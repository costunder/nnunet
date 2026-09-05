"""Run one explicitly selected curriculum arm with strict pre-launch resume guards."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from custom_trainers.onlinecp_curriculum_contract import verify_curriculum_bank_contract
from custom_trainers.onlinecp_curriculum_policy import validate_curriculum_config, curriculum_config_sha256

TRAINERS = {
    "basic": "nnUNetTrainer_250epochs_OnlineBasicCPCurriculumControl",
    "full": "nnUNetTrainer_250epochs_OnlineHierCPCurriculum",
}


def training_command(args):
    bank = Path(args.bank).resolve()
    config_path = Path(args.curriculum_config).resolve()
    config = validate_curriculum_config(json.loads(config_path.read_text(encoding="utf-8")))
    metadata = json.loads(bank.read_text(encoding="utf-8"))
    name = metadata["dataset_name"]
    identity = verify_curriculum_bank_contract(
        bank, curriculum_sha256=curriculum_config_sha256(config),
        expected_candidate_count=config["candidate_count"], dataset_name=name, nnunet_fold=0)
    if args.configuration != identity["verified_configuration"]:
        raise ValueError("Requested configuration differs from the verified bank preprocessing")
    results = os.environ.get("nnUNet_results")
    if not results:
        raise ValueError("Set nnUNet_results to the NEW experiment result root")
    plans_path = Path(identity["files"]["plans"]["path"])
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    if args.configuration not in plans["configurations"]:
        raise ValueError(f"Configuration missing from verified plans: {args.configuration}")
    trainer = TRAINERS[args.arm]
    folder = Path(results) / name / f"{trainer}__{plans_path.stem}__{args.configuration}" / "fold_0"
    checkpoints = [folder / f"checkpoint_{kind}.pth" for kind in ("final", "latest", "best")]
    if args.resume:
        if not any(path.is_file() for path in checkpoints):
            raise FileNotFoundError(
                f"Resume requested but no final/latest/best checkpoint exists: {folder}. "
                "Training was NOT launched; the nnU-Net fresh-start fallback is forbidden.")
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError(
            f"Existing results are preserved: {folder}. Use explicit verified --resume "
            "or a NEW result root; no fresh run may overwrite them.")
    # Call the console entry function, not the upstream module's developer-only
    # __main__ (nnU-Net 2.6.2 assigns an integer into os.environ there).
    entry = "from nnunetv2.run.run_training import run_training_entry; run_training_entry()"
    command = [sys.executable, "-c", entry, str(metadata["dataset_id"]),
               args.configuration, "0", "-tr", trainer, "-p", plans_path.stem, "-device", args.device]
    if args.resume:
        command.append("--c")
    env = {**os.environ, "ONLINE_CP_BANK": str(bank),
           "ONLINE_CP_CURRICULUM_CONFIG": str(config_path), "ONLINE_CP_SEED": str(args.seed)}
    return command, env, folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--curriculum-config", required=True)
    parser.add_argument("--arm", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command, env, folder = training_command(args)
    print(f"[CurriculumRun] arm={args.arm} resume={args.resume} output={folder}")
    print("[Command] " + subprocess.list2cmdline(command))
    if not args.dry_run:
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
