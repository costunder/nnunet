"""Publish existing preparation/CUDA DEBUG evidence without running a model.

The collector uses only the standard library. It verifies source and artifact
bytes, hashes existing lossless graph files for the hardlink receipt, and copies
small metadata only. No tensors, CT, graph payloads or checkpoints are loaded,
copied, repaired, re-created or promoted to production quality evidence.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MAX_METADATA_BYTES = 16 * 1024**2  # Artifact metadata guard, never a data/subset cap.
ALLOWED_EXECUTION_CHANGES = frozenset((
    "hiercp_v1x/transition_v1_data.py", "hiercp_v1x/transition_preparation_storage.py",
    "tools/run_v17_crossed_training.py", "tools/reuse_v17_D_preparation.py"))
ASSIGNMENT_KEYS = ("id", "case_id", "patient_group", "component", "center", "target",
                   "donor_case_id", "donor_component", "donor_group")
MODULES = ("L0", "L1", "L2", "L2_updates")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def real(path, *, directory=False):
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), "Symlink artifact refused: " + str(path))
    path = path.resolve(strict=True)
    require(path.is_relative_to(ROOT), "Artifact must belong to the current workspace")
    require(path.is_dir() if directory else path.is_file(), "Missing artifact: " + str(path))
    return path


def member(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and not PureWindowsPath(relative).is_absolute() and ":" not in relative
            and "\\" not in relative and not any(part in ("", ".", "..") for part in relative.split("/")),
            "Unsafe metadata member: " + str(relative))
    path = real(Path(root) / relative)
    require(path.is_relative_to(Path(root).resolve()), "Member escapes its artifact root")
    return path


def read(path):
    path = real(path)
    require(path.suffix in (".json", ".jsonl") and path.stat().st_size <= MAX_METADATA_BYTES,
            "Only small JSON/JSONL metadata is read: " + str(path))

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    text = path.read_text(encoding="utf8")
    parse = lambda line: json.loads(line, object_pairs_hook=unique, parse_constant=invalid)
    return [parse(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else parse(text)


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            "Expected positive finite measurement: " + name)
    return value


def validate_sources(manifest, previous):
    require(manifest.get("arm") == "D" and manifest.get("debug") is True,
            "Actual D DEBUG identity is required")
    sources = manifest.get("sources")
    require(isinstance(sources, dict) and len(sources) >= 176, "Missing complete execution source inventory")
    for relative, digest in sources.items():
        require(sha(member(ROOT, relative)) == digest, "Current CUDA execution source mismatch: " + relative)
    old = previous["sources"]
    changes = {name: dict(previous_sha256=old.get(name), current_sha256=sources.get(name))
               for name in old.keys() | sources.keys() if old.get(name) != sources.get(name)}
    require(set(changes) == ALLOWED_EXECUTION_CHANGES,
            "Only the measured preparation/storage continuation may change CUDA execution sources")
    require(manifest["scope"] == previous["scope"] and manifest["source"] == previous["source"],
            "Original model/source/scope must be preserved")
    snapshot_proof = manifest["source"]
    snapshot = real(snapshot_proof["source"], directory=True)
    archive = member(ROOT, "versions/v1/pipeline_v1_source.zip")
    require(sha(archive) == snapshot_proof["archive_sha256"], "Original archive byte mismatch")
    checked = {}
    with zipfile.ZipFile(archive) as package:
        entries = [entry for entry in package.infolist() if not entry.is_dir()]
        require(len(entries) == 202 and len({entry.filename for entry in entries}) == 202,
                "The complete original 202-file source archive is required")
        for entry in entries:
            expected = hashlib.sha256(package.read(entry)).hexdigest()
            require(sha(member(snapshot, entry.filename)) == expected,
                    "Preserved original snapshot differs: " + entry.filename)
            checked[entry.filename] = expected
    require(all(checked.get(name) == digest for name, digest in snapshot_proof["verified_files"].items()),
            "Original runtime source proof differs from the archive")
    return dict(current_execution_sources_equal=True, execution_source_count=len(sources),
                execution_source_sha256=sources, changed_source_hashes=changes,
                unchanged_local_module_sha256=sources["hiercp_v1x/transition_v1_local.py"],
                unchanged_frozen_configuration_sha256=sources["config/v17_crossed_training.json"],
                original_archive_sha256=sha(archive), original_snapshot_files=202,
                original_snapshot_file_sha256=checked)


def validate_benchmark(report, summary, manifest):
    require(report.get("debug") is True and report.get("complete") is True
            and report.get("original_inputs_preserved") is True,
            "A completed actual-cache DEBUG benchmark is required")
    for key in ("raw_CT_geometry_rebuilt", "GPU_neural_run", "training_started", "production_ready", "quality_verified"):
        require(report.get(key) is False, "The benchmark must not claim " + key)
    scope = report["scope_contract"]
    require(scope["margin_mm"] == 10 and scope["context_outer_radius_mm"] == 10
            and scope["contract_sha256"] == manifest["scope"] and scope["CNN_shape"] == [48, 48, 48],
            "Benchmark must retain the actual original 10 mm graph/CNN scope")
    for relative, digest in report["source_files"].items():
        require(sha(member(ROOT, relative)) == digest, "Benchmark source mismatch: " + relative)
    index = real(report["input_index"])
    require(sha(index) == report["input_index_sha256"], "Benchmark input cache metadata changed")
    old, new, parity = report["old_serial"], report["new_parallel16"], report["parity"]
    require(old["complete"] is True and new["complete"] is True and parity["complete"] is True,
            "Incomplete benchmark branch")
    require(old["actual_observations"] == new["actual_observations"] == parity["actual_observations"] == 32
            and old["actual_sampled_graphs"] == new["actual_sampled_graphs"] == 64,
            "Both benchmark branches must contain the same 32 genuine two-view observations")
    require(old["workers"] == 1 and new["workers"] == 16
            and old["source_encode_calls"] == 32 and new["source_encode_calls"] == 1,
            "Actual benchmark execution/serialization counters differ")
    require(len(old["records"]) == len(new["records"]) == len(parity["records"]) == 32,
            "Incomplete byte parity records")
    for before, after, matched in zip(old["records"], new["records"], parity["records"]):
        require(before["observation_id"] == after["observation_id"] == matched["observation_id"]
                and before["actual_sampled_measurements"] == after["actual_sampled_measurements"]
                and before["stored"] == after["stored"]
                and matched["byte_exact_parity"] is True and matched["sampled_N_E_parity"] is True,
                "Lossless graph bytes or sampled N/E differ between measured branches")
    for row in report["memory_accounting"]["measurements"]:
        require(row["physical_byte_parity"] is True and row["actual_cached_observations"] in (128, 256, 537),
                "Resident storage accounting parity failed")
        positive(row["canonical_physical_bytes"], "canonical physical bytes")
        positive(row["old_scan_seconds_mean"], "old scan time")
        positive(row["new_incremental_report_seconds_mean"], "incremental report time")
    ratio = positive(old["wall_seconds"], "old wall") / positive(new["wall_seconds"], "new wall")
    require(math.isclose(ratio, report["post_stage_improvement_ratio"], rel_tol=1e-9)
            and math.isclose(ratio, summary["post_stage_improvement_ratio"], rel_tol=1e-9)
            and summary["actual32_byte_SHA_and_sampled_N_E_parity"] is True,
            "Benchmark summary or measured speed ratio differs")
    return dict(actual_post_build_observations=32, actual_sampled_graphs=64,
                compressed_bytes_SHA_and_sampled_N_E_parity=True,
                old_serial_wall_seconds=old["wall_seconds"], new_parallel16_wall_seconds=new["wall_seconds"],
                measured_post_stage_ratio=ratio, source_encode_calls=dict(old=32, new=1),
                resident_measurements=report["memory_accounting"]["measurements"],
                resources=report["resources"], limitations=report["limitations"])


def validate_cuda(root, manifest):
    models, resources = read(root / "model_execution.jsonl"), read(root / "resources.jsonl")
    require(models and resources and all(model == models[0] for model in models), "Constructed model receipt changed")
    model = models[0]
    require(model["arm"] == "D" and model["debug"] is True
            and model["total_parameters"] == model["trainable_parameters"] == 6433126,
            "Actual complete D model receipt is required")
    config = model["local_architecture"]
    require(config["local_layers"] == 3 and config["hidden_dim"] == 128 and config["heads"] == 4
            and config["dense_base_channels"] == 12 and config["dense_feature_dim"] == 32
            and config["checkpoint_local_blocks"] is False and config["checkpoint_dense_encoder"] is False,
            "Original D local model architecture changed")
    updates = read(root / "training" / "updates.jsonl")
    neural = read(root / "training" / "DEBUG_neural_execution.jsonl")
    require(len(updates) == 2 and [row["step"] for row in updates] == [1, 2]
            and len(neural) == 1, "Exactly two successful DEBUG optimizer updates are required")
    result = neural[0]
    require(result.get("actual_CT") is True and result.get("actual_CUDA") is True and result["updates"] == 2
            and result["initial_neural_sha256"] != result["final_neural_sha256"],
            "Actual CT/CUDA weight updates were not verified")
    for module in MODULES:
        positive(result["optimizer_weight_changes"][module], module + " optimizer weight change")
    for row in updates:
        require(row["physical_observations"] == 2 and row["local_graphs"] == 4,
                "DEBUG physical batch or genuine two-view graph count changed")
        gradients = row["gradients"]
        require(gradients["trainable_parameter_tensors"] == gradients["parameter_tensors_with_gradient"] == 677
                and gradients["missing_parameter_gradients"] == [], "A trainable D parameter lacks gradients")
        for module in MODULES:
            positive(gradients[module], module + " gradient")
        positive(row["seconds"], "update time")
        require(row["peak_bytes"] <= resources[-1]["cuda_limit_bytes"], "Update exceeds explicit CUDA budget")
    evaluation_path = real(result["evaluation_artifact"])
    require(evaluation_path.parent == root / "training", "Evaluation belongs to another run")
    evaluation = read(evaluation_path)
    require(evaluation["task"] == "native_observed_P_vs_unobserved_U"
            and evaluation["GT_is_donor_compatibility"] is False
            and evaluation["original_eight_candidate_metrics"] is False
            and evaluation["full_128_U_per_case"] is True,
            "Whole native P/128U GT/evaluation contract changed")
    require((evaluation["cohort"]["records"], evaluation["cohort"]["observed_P"],
             evaluation["cohort"]["unobserved_U"], evaluation["support_records"],
             evaluation["denominators"]["P_U_pairs"]) == (136, 8, 128, 401, 1024),
            "DEBUG whole-case/support coverage differs")
    scoring = evaluation["scoring_contract"]
    require(scoring["l0_only_chunking"] is True and scoring["upper_chunking"] is False
            and scoring["upper_execution"] == "single_joint_case" and scoring["upper_invocations"] == 1
            and scoring["query_GT_in_forward"] is False and len(set(scoring["scored_record_ids"])) == 136,
            "Whole-case upper execution or query label isolation changed")
    require(evaluation["execution_contract_bound"] is True and evaluation["callback_internal_execution_verified"] is True
            and evaluation["raw_CT_execution_verified"] is True and evaluation["C_curriculum_support_reused"] is False
            and evaluation["model_sha256"] == result["final_neural_sha256"]
            and evaluation["metrics"] == result["metrics"], "Actual evaluation is not bound to the updated D model")
    for value in (model, result, evaluation):
        require(value.get("quality_verified") is False and value.get("full_evaluation") is False,
                "DEBUG smoke cannot claim quality or full evaluation")
    require(result.get("full_training") is False and evaluation.get("full_training_complete") is False,
            "DEBUG smoke cannot claim full training")
    return dict(actual_CT=True, actual_CUDA=True, successful_optimizer_updates=2,
                trainable_parameter_tensors=677, parameter_tensors_with_gradient=677,
                optimizer_weight_changes=result["optimizer_weight_changes"],
                initial_neural_sha256=result["initial_neural_sha256"], final_neural_sha256=result["final_neural_sha256"],
                whole_DEBUG_case_observations=136, observed_P=8, unobserved_U=128, refreshed_train_support=401,
                upper_invocations=1, metrics_are_smoke_only=result["metrics"], hardware=resources[-1],
                calibration=read(root / "training" / "calibration.jsonl")), evaluation_path


def validate_reuse(report, manifest):
    require(report["format"] == "v17_D_verified_preparation_import_v1"
            and (report["expected_observations"], report["imported_completed_observations"],
                 report["imported_file_count"], report["pending_observations"]) == (537, 537, 541, 0)
            and report["imported_native_ordinals"] == list(range(537)),
            "Complete actual 537-row/541-file storage reuse receipt is required")
    for key in ("graph_deserialization", "graph_reconstruction", "existing_outputs_modified",
                "uncommitted_rows_imported", "training_or_checkpoint_imported", "all_rows_complete_here"):
        require(report[key] is False, "Storage reuse must not claim or perform " + key)
    require(report["graph_bytes_copied"] == 0 and report["publication_requires_provider_preflight"] is True,
            "Storage import is not provider admission or graph copying")
    source, destination = real(report["source"], directory=True), real(report["destination"], directory=True)
    require(read(destination / "reuse_receipt.json") == {key: value for key, value in report.items()
                                                        if key != "import_wall_seconds"},
            "Published hardlink receipt differs")
    positive(report["import_wall_seconds"], "actual hardlink probe wall time")
    request_path = source / "prepare_request.json"
    require(sha(request_path) == report["source_request_sha256"]
            and sha(source.parent / "manifest.json") == report["source_manifest_sha256"]
            and report["input_inventory_sha256"] == manifest["native_inventory_sha256"],
            "Hardlink reuse source metadata byte binding differs")
    request = read(request_path)
    require(read(destination / "prepare_request.json") == request and request["prepared_observations"] == 537,
            "Destination request changed")
    local = request["local_identity"]
    contract = local["contract"]
    require(local["module_sha256"] == manifest["sources"]["hiercp_v1x/transition_v1_local.py"]
            and contract["ROI_margin_mm"] == 10 and contract["dense_input_shape"] == [5, 48, 48, 48]
            and contract["local_layers"] == 3 and contract["width"] == 128 and contract["views_per_observation"] == 2
            and contract["query_GT_in_forward"] is False and contract["native_comparisons_per_recipient"] == 128
            and request["base"]["graph"]["adaptive_roi_margin_mm"] == 10,
            "Hardlink source must retain unchanged original 10 mm/model/GT contracts")
    files = {}
    for ordinal in range(537):
        name = f"completed/{ordinal:06d}.json"
        old_receipt, new_receipt = member(source, name), member(destination, name)
        require(sha(old_receipt) == sha(new_receipt), "Imported signed receipt bytes changed: " + name)
        row = read(old_receipt)
        for value, size_key in ((row, "compressed_graph_bytes"), (row["shared_source"], "compressed_shared_source_bytes")):
            relative = row["segment"] + "/" + value["path"]
            if relative in files:
                require(files[relative]["sha256"] == value["sha256"], "Contradictory shared-source digest")
                continue
            old_file, new_file = member(source, relative), member(destination, relative)
            require(old_file.samefile(new_file) and old_file.stat().st_size == row[size_key]
                    and sha(old_file) == value["sha256"], "Actual imported graph is not a byte-matched hardlink")
            files[relative] = dict(sha256=value["sha256"], size=old_file.stat().st_size, same_physical_file=True)
    require(len(files) == 541 and sum(item["size"] for item in files.values()) == report["referenced_compressed_bytes"],
            "Imported file count/storage byte total differs")
    return dict(actual_imported_observations=537, actual_hardlinked_files=541,
                all541_file_SHA_size_and_hardlink_identity_verified=True,
                all537_signed_metadata_receipts_byte_equal=True, graph_bytes_copied=0,
                graph_deserialization=False, graph_reconstruction=False, original_source_preserved=True,
                provider_preflight_still_required=True, files=files), destination


def validate_units(path):
    path = real(path)
    require(path.stat().st_size <= MAX_METADATA_BYTES, "UNIT log is unexpectedly large")
    text = path.read_text(encoding="utf8", errors="strict")
    require(re.search(r"Ran 157 tests in [0-9.]+s\s+OK \(skipped=1\)\s*\Z", text) is not None,
            "The final UNIT log must explicitly contain 157 tests, OK, one skip")
    require(len(re.findall(r"\.\.\. ok\s*$", text, re.MULTILINE))
            + len(re.findall(r"^ok\s*$", text, re.MULTILINE)) == 156
            and len(re.findall(r"\.\.\. skipped ", text)) == 1,
            "UNIT log statuses do not establish 156 passes and one skip")
    modules = set(re.findall(r"\((test_[A-Za-z0-9_]+)\.[^)]+\)", text))
    require(modules, "UNIT test module binding is missing")
    sources = {"tests/" + name + ".py": sha(member(ROOT, "tests/" + name + ".py")) for name in sorted(modules)}
    for relative in sources:
        ast.parse(member(ROOT, relative).read_text(encoding="utf8"), filename=relative)
    return dict(tests_run=157, tests_passed=156, tests_skipped=1, tests_failed=0,
                log_sha256=sha(path), test_source_sha256=sources,
                source_binding_scope="Current frozen test sources and log bytes; no GPU quality inference",
                stdlib_AST_test_source_files=len(sources))


def validate_static(path):
    report = read(path)
    require(report.get("static_only") is True and report["python_files"] == len(report["files"]) == 18
            and report["shell_LF_only"] is True and report["bash_syntax_checked"] is True
            and report["training_started"] is False, "Actual final static receipt is required")
    for row in report["files"]:
        file = member(ROOT, row["path"])
        require(row["AST"] is True and sha(file) == row["sha256"], "Static source receipt changed")
        ast.parse(file.read_text(encoding="utf8"), filename=row["path"])
    shell = member(ROOT, "tools/server_v17_crossed.sh")
    require(b"\r" not in shell.read_bytes() and sha(shell) == report["shell_sha256"], "Final Linux shell bytes changed")
    return report


def validate_admitted_reuse(path, manifest):
    report = read(path)
    require(report["format"] == "actual_r3_admitted_cache_STORAGE_reference_DEBUG_v1"
            and report["debug"] is True and report["storage_only"] is True
            and report["whole_import_completed_observations"] == 537 and report["imported_files"] == 541,
            "Actual admitted-cache STORAGE reference receipt is required")
    for key in ("GPU_execution", "full_CLI_neural_execution", "new_neural_run", "new_production_run", "graph_deserialization"):
        require(report[key] is False, "STORAGE-only receipt must not claim " + key)
    require(report["raw_graph_reconstruction_calls"] == report["bytes_written_by_preflight"] == report["graph_bytes_copied"] == 0
            and report["all_original_records_exact"] is True and report["all_original_sampled_view_counts_exact"] is True
            and report["same_command_component_resume_verified"] is True and report["original_index_and_graph_bytes_preserved"] is True,
            "Provider admission/resume or preserved source verification failed")
    source = real(report["source_canonical_cache"], directory=True)
    require(sha(source / "index.json") == report["source_index_sha256"], "Admitted source index changed")
    admission_manifest = real(report["actual_neural_admission_manifest"])
    require(sha(admission_manifest) == report["actual_neural_admission_manifest_sha256"], "Actual r3 admission identity changed")
    previous = read(admission_manifest)
    expected = dict(path=str(source / "index.json"), sha256=report["source_index_sha256"], read_only=True)
    require(previous["reused_prepared_cache"] == report["actual_neural_run_reused_prepared_cache"] == expected,
            "Actual neural run did not admit the preserved cache index")
    changes = {name for name in previous["sources"].keys() | manifest["sources"].keys()
               if previous["sources"].get(name) != manifest["sources"].get(name)}
    require(changes == ALLOWED_EXECUTION_CHANGES, "Admitted-cache continuation changed non-preparation model sources")
    root = real(report["output"], directory=True)
    current_index = root / "new_storage_run" / "canonical_cache" / "index.json"
    require(sha(current_index) == report["new_index_sha256"], "Provider-completed reuse index changed")
    before, after = read(source / "index.json"), read(current_index)
    require(after["complete"] is True and len(after["records"]) == 537
            and before["records"] == after["records"], "Provider preflight altered an actual canonical record")
    require(read(root / "admitted_reference" / "manifest.json") == previous
            and read(root / "admitted_reference" / "STORAGE_REFERENCE_PROVENANCE.json")["storage_only"] is True,
            "Storage reference is not an explicitly declared copy of actual r3 admission")
    for key in ("import_wall_seconds", "preflight_wall_seconds", "same_command_component_resume_seconds"):
        positive(report[key], key)
    return report, root


def publish(output, artifacts, verification, generated):
    output = Path(output).absolute()
    require(output.is_relative_to(ROOT) and not output.exists() and not output.is_symlink(),
            "Evidence output must be a new owned workspace directory; no overwrite")
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for relative, source in artifacts.items():
        source = real(source)
        require(source.suffix in (".json", ".jsonl", ".log") and source.stat().st_size <= MAX_METADATA_BYTES,
                "Only small metadata/log artifacts may be published")
        before = sha(source)
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(source.read_bytes())
        require(sha(dest) == before == sha(source), "Evidence changed while being copied")
        inventory[relative] = dict(sha256=before, size=dest.stat().st_size,
                                  copied_from=str(source), payload_or_checkpoint=False)
    for relative, value in {**generated, "verification.json": verification}.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        inventory[relative] = dict(sha256=sha(path), size=path.stat().st_size, generated_metadata=True)
    with (output / "artifact_inventory.json").open("x", encoding="utf8") as stream:
        json.dump(dict(format="v17_preparation_metadata_inventory_v1", files=inventory,
                       self_excluded=True, no_CT_tensor_graph_checkpoint_payloads_copied=True),
                  stream, indent=2, allow_nan=False)
    return output / "verification.json"


def collect(args):
    benchmark, cuda = real(args.benchmark, directory=True), real(args.cuda, directory=True)
    reuse_path, unit_path = real(args.reuse_report), real(args.unit_log)
    previous_path = real(args.previous_manifest)
    manifest, previous = read(cuda / "manifest.json"), read(previous_path)
    source_proof = validate_sources(manifest, previous)
    report, summary = read(benchmark / "report.json"), read(benchmark / "summary.json")
    benchmark_proof = validate_benchmark(report, summary, manifest)
    cuda_proof, evaluation = validate_cuda(cuda, manifest)
    reuse_proof, reuse_root = validate_reuse(read(reuse_path), manifest)
    unit_proof = validate_units(unit_path)
    static_path, admitted_path = real(args.static_report), real(args.admitted_reuse_report)
    static_proof = validate_static(static_path)
    admitted_proof, admitted_root = validate_admitted_reuse(admitted_path, manifest)
    artifacts = {
        "benchmark/report.json": benchmark / "report.json", "benchmark/summary.json": benchmark / "summary.json",
        "reuse/actual_storage_probe.json": reuse_path, "reuse/reuse_receipt.json": reuse_root / "reuse_receipt.json",
        "reuse/prepare_request.json": reuse_root / "prepare_request.json", "units/final_UNIT.log": unit_path,
        "previous/manifest.json": previous_path, "static/verification.json": static_path,
        "admitted_reuse/actual_admitted_reuse_verification.json": admitted_path,
        "admitted_reuse/STORAGE_REFERENCE_PROVENANCE.json": admitted_root / "admitted_reference" / "STORAGE_REFERENCE_PROVENANCE.json",
        "admitted_reuse/provider_index.json": admitted_root / "new_storage_run" / "canonical_cache" / "index.json",
        "admitted_reuse/reuse_receipt.json": admitted_root / "new_storage_run" / "canonical_cache" / "reuse_receipt.json"}
    for relative in ("manifest.json", "resources.jsonl", "model_execution.jsonl"):
        artifacts["D/" + relative] = cuda / relative
    for path in sorted((cuda / "training").iterdir()):
        if path.suffix in (".json", ".jsonl"):
            artifacts["D/training/" + path.name] = path
    require(evaluation.name in {Path(relative).name for relative in artifacts}, "Final actual evaluation was not selected")
    verification = dict(format="v17_preparation_execution_verification_v1", completed=True,
        collected_at=datetime.now(timezone.utc).isoformat(), debug=True,
        scope="Lossless execution optimization, actual-cache CPU benchmark, actual CT/CUDA two-update D smoke, read-only reuse",
        collector_sha256=sha(Path(__file__)), source_proof=source_proof,
        benchmark=benchmark_proof, actual_CT_CUDA=cuda_proof,
        actual_storage_reuse={key: value for key, value in reuse_proof.items() if key != "files"}, units=unit_proof,
        actual_admitted_reuse_STORAGE_only=admitted_proof, static_checks=static_proof,
        full14102_preparation_completed_by_this_verification=False,
        server_hardware_or_throughput_verified=False, production_batch32_verified=False,
        full_training=False, full_evaluation=False, quality_verified=False, production_ready=False,
        long_training_started=False, original_inputs_and_outputs_preserved=True,
        limitations=["CPU post-build measurement excludes raw CT graph construction and does not estimate server total time",
                     "Two CUDA updates and one whole P/128U case verify mechanics, not ranking quality",
                     "537 DEBUG observations are storage-reused; full14102/server batch32/full21/40 epochs remain unexecuted",
                     "Hardlink import remains separate from provider preflight and checkpoint resume"])
    return publish(args.output, artifacts, verification,
                   {"reuse/hardlink_file_verification.json": reuse_proof, "units/source_binding.json": unit_proof})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="work/v17_preparation_execution_DEBUG_20261005")
    parser.add_argument("--cuda", default="work/v17_fastprep_D_cuda_DEBUG_20261005_r2")
    parser.add_argument("--reuse-report", default="work/v17_actual_storage_reuse_DEBUG_20261005_143301/actual_storage_probe.json")
    parser.add_argument("--unit-log", default="work/v17_fastprep_UNIT_20261005_r2.log")
    parser.add_argument("--previous-manifest", default="validation/v17_crossed_training_20261005/D/manifest.json")
    parser.add_argument("--static-report", default="work/v17_fastprep_static_20261005.json")
    parser.add_argument("--admitted-reuse-report", default="work/v17_admitted_reuse_STORAGE_DEBUG_20261005_144456/actual_admitted_reuse_verification.json")
    parser.add_argument("--output", default="validation/v17_preparation_execution_20261005")
    result = collect(parser.parse_args())
    print(json.dumps(dict(verification=str(result), sha256=sha(result),
                         actual_CT_CUDA_smoke=True, tests_passed=156, tests_skipped=1,
                         full_training=False, full_evaluation=False, quality_verified=False)))


if __name__ == "__main__":
    main()
