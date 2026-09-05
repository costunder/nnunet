#!/usr/bin/env python3
"""nnU-Net v2 downstream study for hierarchical liver-tumor Copy-Paste.

One combined nnU-Net dataset is used with 10 custom splits:
  folds 0-4: baseline, original training cases only
  folds 5-9: matched Copy-Paste arm, original + synthetic training cases
Validation always consists of original cases only. Synthetic cases from a
validation patient are never included in that fold's training set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

VERSION = "hiercp_nnunetv2_small_tumor_v8"
SYN_SUFFIX = "__hpyg_small"


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    case_id: str
    image: Path
    label: Path


@dataclass(frozen=True)
class Synthetic:
    source_id: str
    case_id: str
    image: Path
    label: Path
    voxels: int
    volume_mm3: float
    diameter_mm: float
    selected: bool
    reason: str


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    preprocessed: Path
    results: Path
    dataset_name: str
    dataset_dir: Path
    preprocessed_dir: Path
    model_dir: Path
    synthetic_csv: Path
    splits_json: Path
    train_csv: Path
    eval_dir: Path
    logs: Path


def natural_key(text: str) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"Missing JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise PipelineError(f"JSON root must be an object: {path}")
    return obj


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def discover(root: Path) -> list[Case]:
    images = root / "image"
    labels = root / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise PipelineError(f"Expected image/ and labels/ under {root}")
    out: list[Case] = []
    for image in sorted(images.glob("*_0000.nii.gz"), key=lambda p: natural_key(p.name)):
        cid = image.name[:-len("_0000.nii.gz")]
        label = labels / f"{cid}.nii.gz"
        if not label.is_file():
            raise PipelineError(f"Missing label for {image}: {label}")
        out.append(Case(cid, image.resolve(), label.resolve()))
    if not out:
        raise PipelineError(f"No NIfTI pairs under {root}")
    return out


def load_3d(path: Path, dtype: Any) -> tuple[Any, np.ndarray]:
    nii = nib.load(str(path))
    view = nii
    if len(view.shape) == 4:
        if view.shape[-1] <= 4:
            view = view.slicer[..., 0]
        elif view.shape[0] <= 4:
            view = view.slicer[0, ...]
        else:
            raise PipelineError(f"Cannot infer channel axis: {path} {view.shape}")
    if len(view.shape) != 3:
        raise PipelineError(f"Expected 3D NIfTI: {path} {view.shape}")
    return view, np.asarray(view.dataobj, dtype=dtype, order="C")


def eq_diameter(volume_mm3: float) -> float:
    return 0.0 if volume_mm3 <= 0 else float(2 * (3 * volume_mm3 / (4 * math.pi)) ** (1 / 3))


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src.resolve(), dst)
    else:
        raise PipelineError(f"Bad materialization mode: {mode}")


def dataset_name(cfg: Mapping[str, Any]) -> str:
    did = int(cfg["dataset"]["id"])
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(cfg["dataset"]["name"])).strip("_")
    if not (1 <= did <= 999 and name):
        raise PipelineError("Invalid dataset id/name")
    return f"Dataset{did:03d}_{name}"


def paths_for(workspace: Path, cfg: Mapping[str, Any], override: Path | None) -> Paths:
    root = override.resolve() if override else (workspace / cfg["runtime"].get("nnunet_root", "nnunetv2")).resolve()
    name = dataset_name(cfg)
    raw, prep, results = root / "nnUNet_raw", root / "nnUNet_preprocessed", root / "nnUNet_results"
    trainer = cfg["dataset"]["trainer"]
    plans = cfg["dataset"]["plans"]
    configuration = cfg["dataset"]["configuration"]
    return Paths(
        root, raw, prep, results, name, raw / name, prep / name,
        results / name / f"{trainer}__{plans}__{configuration}",
        root / "synthetic_cases.csv", root / "study_splits.json",
        root / "training_manifest.csv", root / "evaluation", root / "logs"
    )


def validate_cfg(cfg: Mapping[str, Any]) -> None:
    for k in ("dataset", "small_tumor", "preprocess", "training", "runtime"):
        if k not in cfg:
            raise PipelineError(f"Missing config section: {k}")
    if int(cfg["training"].get("num_folds", 5)) != 5:
        raise PipelineError("num_folds must be 5")
    low = float(cfg["small_tumor"]["augmentation_min_equivalent_diameter_mm"])
    high = float(cfg["small_tumor"]["augmentation_max_equivalent_diameter_mm"])
    if high <= low:
        raise PipelineError("Invalid augmentation diameter range")


def analyze_synthetic(original: Case, aug: Case, cfg: Mapping[str, Any]) -> Synthetic:
    onii, olab = load_3d(original.label, np.int16)
    anii, alab = load_3d(aug.label, np.int16)
    if olab.shape != alab.shape or not np.allclose(onii.affine, anii.affine, atol=1e-5):
        raise PipelineError(f"Geometry mismatch: {original.case_id}")
    tumor = int(cfg["small_tumor"].get("tumor_label", 2))
    new = (alab == tumor) & (olab != tumor)
    cc, n = ndi.label(new, structure=np.ones((3, 3, 3), np.uint8))
    voxels = int(new.sum())
    volume = float(voxels * np.prod(onii.header.get_zooms()[:3]))
    diameter = eq_diameter(volume)
    low = float(cfg["small_tumor"]["augmentation_min_equivalent_diameter_mm"])
    high = float(cfg["small_tumor"]["augmentation_max_equivalent_diameter_mm"])
    selected = bool(n > 0 and low < diameter <= high)
    reason = "selected" if selected else ("no_new_tumor" if n == 0 else "outside_diameter_range")
    return Synthetic(original.case_id, original.case_id + SYN_SUFFIX, aug.image, aug.label,
                     voxels, volume, diameter, selected, reason)


def read_synthetic_csv(path: Path) -> list[Synthetic]:
    if not path.is_file():
        return []
    rows: list[Synthetic] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(Synthetic(
                r["source_id"], r["case_id"], Path(r["image"]).resolve(), Path(r["label"]).resolve(),
                int(r["voxels"]), float(r["volume_mm3"]), float(r["diameter_mm"]),
                bool(int(r["selected"])), r["reason"]
            ))
    return rows


def write_synthetic_csv(path: Path, records: Sequence[Synthetic]) -> None:
    fields = ("source_id", "case_id", "selected", "reason", "voxels", "volume_mm3", "diameter_mm", "image", "label")
    atomic_csv(path, [{
        "source_id": r.source_id, "case_id": r.case_id, "selected": int(r.selected),
        "reason": r.reason, "voxels": r.voxels, "volume_mm3": f"{r.volume_mm3:.6f}",
        "diameter_mm": f"{r.diameter_mm:.6f}", "image": str(r.image), "label": str(r.label)
    } for r in records], fields)


def file_sig(path: Path) -> tuple[str, int, int]:
    s = path.resolve().stat()
    return str(path.resolve()), int(s.st_size), int(s.st_mtime_ns)


def study_fingerprint(originals: Sequence[Case], syn: Sequence[Synthetic], cfg: Mapping[str, Any]) -> str:
    payload = {
        "version": VERSION,
        "dataset": cfg["dataset"], "small_tumor": cfg["small_tumor"],
        "original": [(c.case_id, file_sig(c.image), file_sig(c.label)) for c in originals],
        "synthetic": [(r.source_id, r.case_id, file_sig(r.image), file_sig(r.label), round(r.diameter_mm, 6), r.selected) for r in syn],
        "split_seed": int(cfg["training"].get("split_seed", 42))
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_splits(original_ids: Sequence[str], selected: Sequence[Synthetic], seed: int) -> list[dict[str, list[str]]]:
    ids = np.asarray(sorted(original_ids, key=natural_key), dtype=object)
    shuffled = ids[np.random.default_rng(seed).permutation(len(ids))]
    chunks = [set(map(str, x.tolist())) for x in np.array_split(shuffled, 5)]
    all_ids = set(map(str, ids.tolist()))
    baseline, augmented = [], []
    for val in chunks:
        train = sorted(all_ids - val, key=natural_key)
        val_ids = sorted(val, key=natural_key)
        baseline.append({"train": train, "val": val_ids})
        aug_ids = sorted([r.case_id for r in selected if r.source_id in set(train)], key=natural_key)
        augmented.append({"train": train + aug_ids, "val": val_ids})
    return baseline + augmented


def build_dataset(medical_root: Path, workspace: Path, cfg: Mapping[str, Any], p: Paths,
                  mode: str, overwrite: bool) -> None:
    originals = discover(medical_root / "Data")
    orig_map = {c.case_id: c for c in originals}
    aug_cases = discover(workspace / "output" / "valid")
    cached = [] if overwrite else read_synthetic_csv(p.synthetic_csv)
    cached_map = {r.source_id: r for r in cached}
    syn: list[Synthetic] = []
    total = len(aug_cases)

    for index, aug in enumerate(aug_cases, start=1):
        if aug.case_id not in orig_map:
            raise PipelineError(f"No original for augmented case {aug.case_id}")

        record = cached_map.get(aug.case_id)
        reusable = bool(
            record is not None
            and record.image == aug.image
            and record.label == aug.label
            and record.image.is_file()
            and record.label.is_file()
        )

        if reusable:
            syn.append(record)
            print(
                f"[Reuse] synthetic {index}/{total} {aug.case_id} "
                f"diameter={record.diameter_mm:.2f}mm "
                f"selected={int(record.selected)}",
                flush=True,
            )
            continue

        print(
            f"[Analyze] synthetic {index}/{total} {aug.case_id}",
            flush=True,
        )

        record = analyze_synthetic(orig_map[aug.case_id], aug, cfg)
        cached_map[aug.case_id] = record
        syn.append(record)

        # case 하나가 끝날 때마다 저장한다.
        write_synthetic_csv(
            p.synthetic_csv,
            sorted(
                cached_map.values(),
                key=lambda item: natural_key(item.source_id),
            ),
        )

        print(
            f"[OK] synthetic {index}/{total} {aug.case_id} "
            f"diameter={record.diameter_mm:.2f}mm "
            f"selected={int(record.selected)}",
            flush=True,
        )

    # 현재 validated_data에 포함된 case만 최종 유지한다.
    write_synthetic_csv(p.synthetic_csv, syn)
    selected = [r for r in syn if r.selected]
    if not selected:
        raise PipelineError("No validated synthetic lesion satisfies the small-tumor diameter threshold")
    fp = study_fingerprint(originals, syn, cfg)
    marker = p.dataset_dir / "hiercp_study.json"
    splits = make_splits([c.case_id for c in originals], selected, int(cfg["training"].get("split_seed", 42)))
    split_payload = {"version": VERSION, "fingerprint": fp, "num_base_folds": 5, "splits": splits}
    atomic_json(p.splits_json, split_payload)
    if p.dataset_dir.exists() and not overwrite:
        if not marker.is_file() or load_json(marker).get("fingerprint") != fp:
            raise PipelineError("Existing nnU-Net raw dataset does not match current inputs/config")
        print(f"[Reuse] raw dataset {p.dataset_dir} originals={len(originals)} small_synthetic={len(selected)}")
        return
    if p.dataset_dir.exists():
        shutil.rmtree(p.dataset_dir)
    tmp = p.dataset_dir.with_name(f".{p.dataset_name}.tmp.{os.getpid()}")
    if tmp.exists(): shutil.rmtree(tmp)
    (tmp / "imagesTr").mkdir(parents=True)
    (tmp / "labelsTr").mkdir(parents=True)
    (tmp / "imagesTs").mkdir(parents=True)
    for c in originals:
        materialize(c.image, tmp / "imagesTr" / f"{c.case_id}_0000.nii.gz", mode)
        materialize(c.label, tmp / "labelsTr" / f"{c.case_id}.nii.gz", mode)
    for r in selected:
        materialize(r.image, tmp / "imagesTr" / f"{r.case_id}_0000.nii.gz", mode)
        materialize(r.label, tmp / "labelsTr" / f"{r.case_id}.nii.gz", mode)
    tumor = int(cfg["small_tumor"].get("tumor_label", 2))
    atomic_json(tmp / "dataset.json", {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1, "tumor": tumor},
        "numTraining": len(originals) + len(selected), "file_ending": ".nii.gz"
    })
    atomic_json(tmp / "hiercp_study.json", {
        "version": VERSION, "fingerprint": fp, "original_cases": len(originals),
        "selected_small_synthetic": len(selected), "diameter_max_mm": cfg["small_tumor"]["augmentation_max_equivalent_diameter_mm"]
    })
    tmp.replace(p.dataset_dir)
    print(f"[OK] raw dataset {p.dataset_dir} originals={len(originals)} small_synthetic={len(selected)} total={len(originals)+len(selected)}")


def install_splits(p: Paths) -> None:
    obj = load_json(p.splits_json)
    p.preprocessed_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(p.preprocessed_dir / "splits_final.json", json.dumps(obj["splits"], indent=2) + "\n")
    print(f"[OK] installed controlled 10-fold split file: {p.preprocessed_dir / 'splits_final.json'}")


def nn_env(p: Paths, cfg: Mapping[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["nnUNet_raw"], env["nnUNet_preprocessed"], env["nnUNet_results"] = str(p.raw), str(p.preprocessed), str(p.results)
    env["nnUNet_n_proc_DA"] = str(int(cfg["training"].get("nnunet_n_proc_DA", 8)))
    env.setdefault("OMP_NUM_THREADS", "1"); env.setdefault("MKL_NUM_THREADS", "1"); env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def require_command(name: str) -> str:
    out = shutil.which(name)
    if out is None: raise PipelineError(f"Command not found in active environment: {name}")
    return out


def quote(v: str) -> str:
    import shlex
    return shlex.quote(v)


def run_cmd(cmd: Sequence[str], env: Mapping[str, str], log: Path, cwd: Path, dry: bool) -> None:
    print("\n$ " + " ".join(quote(x) for x in cmd))
    if dry: return
    log.parent.mkdir(parents=True, exist_ok=True)
    lf = None
    try: lf = log.open("a", encoding="utf-8")
    except OSError as exc: print(f"[Warn] log disabled: {exc}")
    proc = subprocess.Popen(list(cmd), cwd=str(cwd), env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if lf:
            try:
                lf.write(line)
            except OSError as exc:
                print(f"[Warn] log write failed; disabling {log}: {exc}")
                try:
                    lf.close()
                except OSError as close_exc:
                    print(f"[Warn] log close failed for {log}: {close_exc}")
                lf = None
    rc = proc.wait()
    if lf:
        try:
            lf.close()
        except OSError as exc:
            print(f"[Warn] log close failed for {log}: {exc}")
    if rc: raise PipelineError(f"Command failed ({rc}): {' '.join(cmd)}")


def plan_ready(p: Paths, cfg: Mapping[str, Any]) -> bool:
    return (p.preprocessed_dir / f"{cfg['dataset']['plans']}.json").is_file() and any(
        cfg["dataset"]["configuration"] in x.name for x in p.preprocessed_dir.glob("*") if x.is_dir()
    )


def plan(project: Path, p: Paths, cfg: Mapping[str, Any], dry: bool) -> None:
    if plan_ready(p, cfg):
        install_splits(p); print(f"[Reuse] preprocessing ready: {p.preprocessed_dir}"); return
    free = shutil.disk_usage(p.root).free / 1024**3
    minimum = float(cfg["runtime"].get("minimum_free_gb_before_preprocess", 80))
    if free < minimum: raise PipelineError(f"Free space {free:.1f} GiB < configured minimum {minimum:.1f} GiB")
    exe = require_command("nnUNetv2_plan_and_preprocess")
    cmd = [exe, "-d", str(int(cfg["dataset"]["id"])), "-pl", cfg["dataset"]["planner"], "-c", cfg["dataset"]["configuration"]]
    if cfg["preprocess"].get("verify_dataset_integrity", True): cmd.append("--verify_dataset_integrity")
    if cfg["preprocess"].get("no_progress_bar", True): cmd.append("--no_pbar")
    if cfg["preprocess"].get("processes") is not None: cmd += ["-np", str(int(cfg["preprocess"]["processes"]))]
    run_cmd(cmd, nn_env(p, cfg), p.logs / "plan_and_preprocess.log", project, dry)
    if not dry:
        if not plan_ready(p, cfg): raise PipelineError("Expected plans/preprocessed data missing after command")
        install_splits(p)


def parse_folds(value: str | None, cfg: Mapping[str, Any]) -> list[int]:
    vals = cfg["training"].get("folds", [0,1,2,3,4]) if not value else [int(x) for x in value.split(",") if x.strip()]
    vals = sorted(set(map(int, vals)))
    if any(x not in range(5) for x in vals): raise PipelineError(f"Base folds must be 0-4: {vals}")
    return vals


def validation_complete(fold_dir: Path, val_ids: Sequence[str]) -> bool:
    vd = fold_dir / "validation"
    return (vd / "summary.json").is_file() and all((vd / f"{x}.nii.gz").is_file() for x in val_ids)


def update_train_csv(path: Path, row: Mapping[str, Any]) -> None:
    fields = ("condition","base_fold","nnunet_fold","status","train_count","val_count","updated_at","message")
    rows = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
    rows = [r for r in rows if not (r["condition"] == str(row["condition"]) and r["base_fold"] == str(row["base_fold"]))]
    rows.append({k: str(row.get(k,"")) for k in fields})
    rows.sort(key=lambda r:(r["condition"],int(r["base_fold"])))
    atomic_csv(path, rows, fields)


def train_one(project: Path, p: Paths, cfg: Mapping[str, Any], condition: str, base_fold: int,
              nnfold: int, split: Mapping[str, Sequence[str]], device: str, dry: bool) -> None:
    if condition == "hierarchical_copy_paste":
        raise PipelineError(
            "The legacy globally generated CP dataset cannot prove that this fold's "
            "validation labels were excluded from GNN fitting/model selection. "
            "Use the fold-specific pairedcp/onlinecp pipeline in a new workspace. "
            "Existing datasets and results have not been changed."
        )
    fold_dir = p.model_dir / f"fold_{nnfold}"
    final = fold_dir / "checkpoint_final.pth"
    val_ids = list(map(str, split["val"]))
    if final.is_file() and validation_complete(fold_dir, val_ids):
        print(f"[Reuse] {condition} base_fold={base_fold} nnfold={nnfold} complete")
        update_train_csv(p.train_csv,{"condition":condition,"base_fold":base_fold,"nnunet_fold":nnfold,"status":"complete","train_count":len(split["train"]),"val_count":len(val_ids),"updated_at":time.strftime('%FT%T'),"message":"reused"})
        return
    exe = require_command("nnUNetv2_train")
    cmd = [exe,str(int(cfg["dataset"]["id"])),cfg["dataset"]["configuration"],str(nnfold),"-tr",cfg["dataset"]["trainer"],"-p",cfg["dataset"]["plans"],"-device",device]
    status = "fresh"
    if final.is_file(): cmd.append("--val"); status = "validation_only"
    elif (fold_dir/"checkpoint_latest.pth").is_file() or (fold_dir/"checkpoint_best.pth").is_file(): cmd.append("--c"); status = "resume"
    if cfg["training"].get("save_npz",False): cmd.append("--npz")
    update_train_csv(p.train_csv,{"condition":condition,"base_fold":base_fold,"nnunet_fold":nnfold,"status":status,"train_count":len(split["train"]),"val_count":len(val_ids),"updated_at":time.strftime('%FT%T'),"message":"started"})
    run_cmd(cmd, nn_env(p,cfg), p.logs/f"{condition}_base{base_fold}_nnfold{nnfold}.log", project, dry)
    if dry: return
    if not final.is_file(): raise PipelineError(f"checkpoint_final.pth missing: {fold_dir}")
    if not validation_complete(fold_dir,val_ids):
        vcmd=[exe,str(int(cfg["dataset"]["id"])),cfg["dataset"]["configuration"],str(nnfold),"-tr",cfg["dataset"]["trainer"],"-p",cfg["dataset"]["plans"],"--val","-device",device]
        run_cmd(vcmd,nn_env(p,cfg),p.logs/f"{condition}_base{base_fold}_nnfold{nnfold}.log",project,False)
    if not validation_complete(fold_dir,val_ids): raise PipelineError(f"Validation incomplete: {fold_dir}")
    update_train_csv(p.train_csv,{"condition":condition,"base_fold":base_fold,"nnunet_fold":nnfold,"status":"complete","train_count":len(split["train"]),"val_count":len(val_ids),"updated_at":time.strftime('%FT%T'),"message":"completed"})


def train(project: Path, p: Paths, cfg: Mapping[str, Any], folds: Sequence[int], device: str, dry: bool) -> None:
    if not plan_ready(p,cfg): raise PipelineError("Preprocessing is not ready")
    install_splits(p)
    splits=load_json(p.splits_json)["splits"]
    for f in folds:
        train_one(project,p,cfg,"baseline",f,f,splits[f],device,dry)
        train_one(project,p,cfg,"hierarchical_copy_paste",f,f+5,splits[f+5],device,dry)


def dice(a: np.ndarray,b: np.ndarray)->float:
    s=int(a.sum()+b.sum())
    return 1.0 if s==0 else float(2*np.logical_and(a,b).sum()/s)


def lesion_metrics(gt: np.ndarray,pred: np.ndarray,spacing: Sequence[float],bins: Sequence[float]) -> tuple[list[dict[str,Any]],dict[str,int]]:
    st=np.ones((3,3,3),np.uint8); gcc,ng=ndi.label(gt,structure=st); pcc,npred=ndi.label(pred,structure=st)
    gs=np.bincount(gcc.ravel(),minlength=ng+1); ps=np.bincount(pcc.ravel(),minlength=npred+1)
    inter=np.zeros((ng,npred),dtype=np.int64)
    m=(gcc>0)&(pcc>0)
    if m.any():
        pairs,c=np.unique(np.stack([gcc[m]-1,pcc[m]-1],1),axis=0,return_counts=True); inter[pairs[:,0],pairs[:,1]]=c
    matches={}
    if ng and npred:
        rr,cc=linear_sum_assignment(-inter)
        matches={int(r):int(c) for r,c in zip(rr,cc) if inter[r,c]>0}
    vv=float(np.prod(spacing)); bs=sorted(map(float,bins))
    def bname(d:float)->str:
        if d<=bs[0]: return f"le_{bs[0]:g}mm"
        if d<=bs[1]: return f"gt_{bs[0]:g}_le_{bs[1]:g}mm"
        return f"gt_{bs[1]:g}mm"
    rows=[]
    for i in range(ng):
        gv=int(gs[i+1]); vol=gv*vv; d=eq_diameter(vol); j=matches.get(i); det=j is not None
        pv=int(ps[j+1]) if det else 0; iv=int(inter[i,j]) if det else 0
        rows.append({"diameter_mm":d,"size_bin":bname(d),"detected":int(det),"lesion_dice":float(2*iv/(gv+pv)) if det else 0.0})
    matched_pred=len(set(matches.values()))
    return rows,{"gt":ng,"pred":npred,"matched_pred":matched_pred,"fp":npred-matched_pred}


def evaluate(p: Paths,cfg: Mapping[str,Any],folds: Sequence[int])->None:
    splits=load_json(p.splits_json)["splits"]; tumor=int(cfg["small_tumor"].get("tumor_label",2)); bins=cfg["small_tumor"].get("evaluation_bins_mm",[10,20])
    cases=[]; lesions=[]
    for f in folds:
        if splits[f]["val"]!=splits[f+5]["val"]: raise PipelineError(f"Validation split mismatch fold {f}")
        bd=p.model_dir/f"fold_{f}"/"validation"; ad=p.model_dir/f"fold_{f+5}"/"validation"
        for cid in splits[f]["val"]:
            refp=p.dataset_dir/"labelsTr"/f"{cid}.nii.gz"; bp=bd/f"{cid}.nii.gz"; ap=ad/f"{cid}.nii.gz"
            if not all(x.is_file() for x in (refp,bp,ap)): raise PipelineError(f"Missing prediction/reference for {cid}")
            rn,r=load_3d(refp,np.int16); _,b=load_3d(bp,np.int16); _,a=load_3d(ap,np.int16)
            gt=r==tumor; bm=b==tumor; am=a==tumor
            bl,bc=lesion_metrics(gt,bm,rn.header.get_zooms()[:3],bins); al,ac=lesion_metrics(gt,am,rn.header.get_zooms()[:3],bins)
            cases.append({"case_id":cid,"fold":f,"baseline_dice":dice(gt,bm),"aug_dice":dice(gt,am),"gt_lesions":bc["gt"],"baseline_pred":bc["pred"],"baseline_fp":bc["fp"],"aug_pred":ac["pred"],"aug_fp":ac["fp"]})
            for i,(x,y) in enumerate(zip(bl,al),1): lesions.append({"case_id":cid,"fold":f,"component":i,"diameter_mm":x["diameter_mm"],"size_bin":x["size_bin"],"baseline_detected":x["detected"],"aug_detected":y["detected"],"baseline_lesion_dice":x["lesion_dice"],"aug_lesion_dice":y["lesion_dice"]})
    if not cases: raise PipelineError("No cases evaluated")
    p.eval_dir.mkdir(parents=True,exist_ok=True)
    atomic_csv(p.eval_dir/"case_metrics.csv",cases,tuple(cases[0]))
    atomic_csv(p.eval_dir/"lesion_metrics.csv",lesions,tuple(lesions[0]) if lesions else ("case_id",))
    def summary(prefix:str)->dict[str,Any]:
        detected=sum(int(x[f"{prefix}_detected"]) for x in lesions); pred=sum(int(x[f"{prefix}_pred"]) for x in cases); fp=sum(int(x[f"{prefix}_fp"]) for x in cases)
        by={}
        for bn in sorted({x["size_bin"] for x in lesions}):
            rr=[x for x in lesions if x["size_bin"]==bn]; d=sum(int(x[f"{prefix}_detected"]) for x in rr); by[bn]={"gt":len(rr),"detected":d,"recall":d/len(rr) if rr else None}
        return {"mean_case_tumor_dice":float(np.mean([float(x[f"{prefix}_dice"]) for x in cases])),"gt_lesions":len(lesions),"detected":detected,"lesion_recall":detected/len(lesions) if lesions else None,"predicted_lesions":pred,"false_positive_lesions":fp,"false_positive_per_case":fp/len(cases),"by_size":by}
    bs,asum=summary("baseline"),summary("aug")
    out={"version":VERSION,"dataset":p.dataset_name,"folds":list(folds),"baseline":bs,"hierarchical_copy_paste":asum,"difference":{"mean_case_tumor_dice":asum["mean_case_tumor_dice"]-bs["mean_case_tumor_dice"],"lesion_recall":asum["lesion_recall"]-bs["lesion_recall"],"false_positive_per_case":asum["false_positive_per_case"]-bs["false_positive_per_case"]}}
    atomic_json(p.eval_dir/"summary.json",out)
    md=["# nnU-Net v2 Small-Tumor Copy-Paste Comparison","",f"- Cases: {len(cases)} original validation cases",f"- Dataset: `{p.dataset_name}`","","| Metric | Baseline | Copy-Paste | Difference |","|---|---:|---:|---:|",f"| Mean tumor Dice | {bs['mean_case_tumor_dice']:.4f} | {asum['mean_case_tumor_dice']:.4f} | {out['difference']['mean_case_tumor_dice']:+.4f} |",f"| Lesion recall | {bs['lesion_recall']:.4f} | {asum['lesion_recall']:.4f} | {out['difference']['lesion_recall']:+.4f} |",f"| FP lesions/case | {bs['false_positive_per_case']:.3f} | {asum['false_positive_per_case']:.3f} | {out['difference']['false_positive_per_case']:+.3f} |","","## Recall by lesion size","","| Bin | GT | Baseline | Copy-Paste |","|---|---:|---:|---:|"]
    for bn in sorted(set(bs["by_size"])|set(asum["by_size"])):
        b0=bs["by_size"].get(bn,{}); a0=asum["by_size"].get(bn,{}); md.append(f"| {bn} | {b0.get('gt',a0.get('gt',0))} | {b0.get('recall',float('nan')):.4f} | {a0.get('recall',float('nan')):.4f} |")
    atomic_text(p.eval_dir/"comparison.md","\n".join(md)+"\n")
    print(f"[OK] evaluation: {p.eval_dir / 'summary.json'}")


def status(p: Paths,cfg: Mapping[str,Any])->None:
    print("nnU-Net v2 small-tumor study status")
    print("  root:             ",p.root); print("  dataset:          ",p.dataset_name)
    marker=p.dataset_dir/"hiercp_study.json"
    if marker.is_file():
        m=load_json(marker); print("  raw dataset:       ready"); print("  original cases:   ",m.get("original_cases")); print("  small synthetic:  ",m.get("selected_small_synthetic"))
    else: print("  raw dataset:       missing")
    print("  preprocessed:     ","ready" if plan_ready(p,cfg) else "missing")
    complete=[0,0]
    if p.splits_json.is_file():
        sp=load_json(p.splits_json)["splits"]
        for f in range(5):
            complete[0]+=int((p.model_dir/f"fold_{f}"/"checkpoint_final.pth").is_file() and validation_complete(p.model_dir/f"fold_{f}",sp[f]["val"]))
            complete[1]+=int((p.model_dir/f"fold_{f+5}"/"checkpoint_final.pth").is_file() and validation_complete(p.model_dir/f"fold_{f+5}",sp[f+5]["val"]))
    print(f"  baseline folds:    {complete[0]}/5"); print(f"  copy-paste folds:  {complete[1]}/5")
    print("  evaluation:       ","ready" if (p.eval_dir/"summary.json").is_file() else "missing")


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target",choices=("check","prepare","plan","train","evaluate","all","status"))
    ap.add_argument("--project-root",required=True); ap.add_argument("--medical-root",required=True); ap.add_argument("--workspace",required=True); ap.add_argument("--config",required=True)
    ap.add_argument("--nnunet-root",default=None); ap.add_argument("--device",default="auto"); ap.add_argument("--folds",default=None); ap.add_argument("--materialization",choices=("symlink","hardlink","copy"),default="symlink"); ap.add_argument("--overwrite",action="store_true"); ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args(); project=Path(a.project_root).resolve(); medical=Path(a.medical_root).resolve(); workspace=Path(a.workspace).resolve(); cfg=load_json(Path(a.config).resolve()); validate_cfg(cfg)
    p=paths_for(workspace,cfg,Path(a.nnunet_root).expanduser() if a.nnunet_root else None)
    for d in (p.root,p.raw,p.preprocessed,p.results,p.logs): d.mkdir(parents=True,exist_ok=True)
    device="cuda" if a.device in ("auto","cuda","cuda:0") or str(a.device).startswith("cuda") else a.device
    folds=parse_folds(a.folds,cfg)
    if a.target=="status": status(p,cfg); return
    if a.target=="check":
        require_command("nnUNetv2_plan_and_preprocess"); require_command("nnUNetv2_train"); print(f"[OK] nnU-Net commands available; free={shutil.disk_usage(p.root).free/1024**3:.1f} GiB; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}"); return
    if a.target in ("prepare","all"): build_dataset(medical,workspace,cfg,p,a.materialization,a.overwrite)
    if a.target in ("plan","all"): plan(project,p,cfg,a.dry_run)
    if a.target in ("train","all"): train(project,p,cfg,folds,device,a.dry_run)
    if a.target in ("evaluate","all"):
        if a.dry_run: print("[Dry-run] evaluation skipped")
        else: evaluate(p,cfg,folds)
    print("\n[Done] nnU-Net v2 small-tumor study completed/requested"); status(p,cfg)


if __name__=="__main__": main()
