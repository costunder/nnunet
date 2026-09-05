"""Strict opt-in launcher for actual nnU-Net difficulty feedback (no legacy resume)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from custom_trainers.onlinecp_curriculum_contract import verify_curriculum_bank_contract
from custom_trainers.onlinecp_feedback_policy import validate_feedback_config, feedback_config_sha256

TRAINERS = {
    "basic": "nnUNetTrainer_250epochs_OnlineBasicCPFeedbackControl",
    "full": "nnUNetTrainer_250epochs_OnlineHierCPFeedback",
}


def training_command(args):
    bank = Path(args.bank).resolve()
    config_path = Path(args.feedback_config).resolve()
    config = validate_feedback_config(json.loads(config_path.read_text(encoding="utf-8")))
    index = json.loads(bank.read_text(encoding="utf-8"))
    identity = verify_curriculum_bank_contract(
        bank, curriculum_sha256=feedback_config_sha256(config),
        expected_candidate_count=128, dataset_name=index["dataset_name"], nnunet_fold=0,
        contract_filename="feedback_contract.json",
    )
    if args.configuration != identity["verified_configuration"]:
        raise ValueError("Requested configuration is not the verified preprocessing configuration")
    results = os.environ.get("nnUNet_results")
    if not results:
        raise ValueError("Set nnUNet_results to the intended new experiment result root")
    plans_path = Path(identity["files"]["plans"]["path"])
    trainer = TRAINERS[args.arm]
    folder = Path(results).resolve() / index["dataset_name"] / f"{trainer}__{plans_path.stem}__{args.configuration}" / "fold_0"
    if args.resume:
        if not any((folder / f"checkpoint_{name}.pth").is_file() for name in ("final", "latest", "best")):
            raise FileNotFoundError(f"No feedback checkpoint to resume: {folder}. Training was NOT launched.")
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Existing results preserved: {folder}. Use verified --resume or a new result root.")
    env = {**os.environ, "ONLINE_CP_BANK": str(bank), "ONLINE_CP_FEEDBACK_CONFIG": str(config_path),
           "ONLINE_CP_SEED": str(args.seed)}
    if args.arm == "full":
        if not args.feedback_gnn_config or not args.feedback_raw_root:
            raise ValueError("Full feedback requires --feedback-gnn-config and --feedback-raw-root")
        gnn_path = Path(args.feedback_gnn_config).resolve()
        raw_root = Path(args.feedback_raw_root).resolve()
        if not gnn_path.is_file() or not (raw_root / "online_cp_dataset.json").is_file():
            raise FileNotFoundError("GNN config or verified nnU-Net raw dataset marker is missing")
        from hiercp.feedback import validate_feedback_gnn_config
        gnn_config = json.loads(gnn_path.read_text(encoding="utf-8"))
        validate_feedback_gnn_config({**gnn_config, "raw_data_root": str(raw_root),
                                      "graph_cache_dir": str(folder / "feedback_graph_cache")})
        env["ONLINE_CP_FEEDBACK_GNN_CONFIG"] = str(gnn_path)
        env["ONLINE_CP_FEEDBACK_RAW_ROOT"] = str(raw_root)
    entry = "from nnunetv2.run.run_training import run_training_entry; run_training_entry()"
    command = [sys.executable, "-c", entry, str(index["dataset_id"]), args.configuration,
               "0", "-tr", trainer, "-p", plans_path.stem, "-device", args.device]
    if args.resume:
        command.append("--c")
    return command, env, folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--feedback-config", required=True)
    parser.add_argument("--feedback-gnn-config")
    parser.add_argument("--feedback-raw-root", help="Verified nnU-Net raw Dataset... directory, not Medical root")
    parser.add_argument("--arm", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command, env, folder = training_command(args)
    print(f"[FeedbackRun] arm={args.arm} resume={args.resume} output={folder}")
    print("[Command] " + subprocess.list2cmdline(command))
    if not args.dry_run:
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
