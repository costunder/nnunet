"""Record an explicitly bound original-v1 foreground training invocation.

Only logging and calibration publication are adapted. Original pipeline
arguments, sampler mathematics, optimizer updates, and signal handling remain
owned by the preserved pipeline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid


FORMAT = "hiercp_v1_epoch_telemetry_request_v1"
REQUEST_FIELDS = {
    "format", "output_dir", "mode", "native_reference", "stage",
    "baseline_manifest_sha256", "baseline_sampling_contract_sha256",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _manifest(path):
    payload = _read(path)
    plain = dict(payload)
    expected = plain.pop("manifest_sha256", None)
    if not _sha256(expected) or _canonical_hash(plain) != expected:
        raise ValueError("Telemetry experiment manifest identity changed")
    return payload


def _verify_contract_files(source_root, contract):
    """Read the bound files, including the recording adapter; never rewrite."""
    source_root = Path(source_root).resolve(strict=True)
    files = contract.get("source_identity", {}).get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Telemetry requires an actual snapshot source inventory")
    for name, expected in files.items():
        target = (source_root / name).resolve(strict=True)
        if (not target.is_relative_to(source_root) or not target.is_file()
                or not _sha256(expected) or _digest(target) != expected):
            raise ValueError(f"Telemetry snapshot source identity changed: {name}")


def verify_baseline_binding(request):
    native = Path(request["native_reference"]).resolve(strict=True)
    manifest = _manifest(native / "manifest.json")
    contract = _read(native / "sampling/v1.0.json")
    plain = dict(contract)
    expected = plain.pop("contract_sha256", None)
    if (manifest.get("manifest_sha256") != request["baseline_manifest_sha256"]
            or manifest.get("sampling_contract", {}).get("mode") != "native"
            or manifest.get("reference_experiment") is not None
            or contract.get("mode") != "native"
            or not _sha256(expected) or _canonical_hash(plain) != expected
            or expected != request["baseline_sampling_contract_sha256"]
            or manifest.get("sampling_contracts", {}).get("v1.0") != contract):
        raise ValueError("Telemetry native reference or sampling identity changed")
    _verify_contract_files(native / "source/v1.0", contract)
    return native, manifest, contract


def validate_request(request, contract_path, contract):
    if (not isinstance(request, dict) or set(request) != REQUEST_FIELDS
            or request["format"] != FORMAT or request["stage"] != "v1.0"
            or request["mode"] not in ("native", "strict_nested")
            or request["mode"] != contract.get("mode")
            or not _sha256(request["baseline_manifest_sha256"])
            or not _sha256(request["baseline_sampling_contract_sha256"])):
        raise ValueError("Explicit original-v1 telemetry request fields differ")
    if any(not isinstance(request[key], str) or not Path(request[key]).is_absolute()
           for key in ("output_dir", "native_reference")):
        raise ValueError("Telemetry output and native reference must be absolute paths")
    contract_path = Path(contract_path).resolve(strict=True)
    if contract_path.name != "v1.0.json" or contract_path.parent.name != "sampling":
        raise ValueError("Telemetry requires the explicit v1.0 sampling contract")
    experiment = contract_path.parent.parent
    output = Path(request["output_dir"]).resolve()
    results = experiment / "results/v1.0"
    if output != results and not output.is_relative_to(results):
        raise ValueError("Telemetry output must belong to this experiment's v1.0 results")
    native, baseline, native_contract = verify_baseline_binding(request)
    current = _manifest(experiment / "manifest.json")
    if current.get("sampling_contracts", {}).get("v1.0") != contract:
        raise ValueError("Telemetry request differs from its experiment sampler")
    if request["mode"] == "native":
        if experiment != native or current != baseline or contract != native_contract:
            raise ValueError("Native telemetry must use its own canonical preparation")
    elif (current.get("reference_experiment") != {
            "path": str(native), "manifest_sha256": request["baseline_manifest_sha256"]}
            or current.get("split") != baseline.get("split")
            or current.get("medical_root") != baseline.get("medical_root")):
        raise ValueError("Nested telemetry must use the same explicit native reference")
    return experiment


def _validate_calibration(native, calibration):
    if not isinstance(calibration, dict):
        raise ValueError("Native telemetry requires measured calibration metadata")
    identity = calibration.get("identity", {})
    batch = calibration.get("selected_batch_size")
    workers = calibration.get("selected_num_workers")
    fingerprint = calibration.get("resource_fingerprint")
    if (calibration.get("format") != "hiercp_preflight_calibration_v2"
            or identity.get("seed") != 42
            or Path(identity.get("cache_dir", "")).resolve() != native / "shared/cache"
            or Path(identity.get("checkpoint_path", "")).resolve()
            != native / "results/v1.0/checkpoint_best.pt"
            or type(batch) is not int or batch <= 0
            or type(workers) is not int or workers < 0
            or not isinstance(fingerprint, dict) or not fingerprint):
        raise ValueError("Native telemetry calibration identity or measured values differ")
    _canonical_hash(calibration)
    return batch, workers, fingerprint


def _native_lock_value(request):
    native, _, _ = verify_baseline_binding(request)
    path = native / "results/v1.0/checkpoint_best.pt.preflight.json"
    calibration = _read(path)
    batch, workers, fingerprint = _validate_calibration(native, calibration)
    value = dict(selected_batch_size=batch, selected_num_workers=workers,
                 baseline_calibration_sha256=_digest(path), resource_fingerprint=fingerprint,
                 baseline_sampling_contract_sha256=request["baseline_sampling_contract_sha256"],
                 baseline_manifest_sha256=request["baseline_manifest_sha256"])
    _canonical_hash(value)  # Reject nonfinite/unserializable calibration metadata.
    return native / "shared/execution_lock.json", value


def verify_execution_lock(request, *, create=False):
    """Publish native measurement once; the nested arm can only verify it."""
    if create and request["mode"] != "native":
        raise ValueError("Nested telemetry cannot publish the native execution lock")
    path, value = _native_lock_value(request)
    if path.exists():
        if _read(path) != value:
            raise ValueError("Native execution lock changed; no overwrite permitted")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        raise ValueError("Measured native execution lock is required before nested training")
    return value


def install_preflight_publication_guard(pipeline, request):
    """Bind the lock immediately after the original atomic preflight write."""
    original = pipeline._write_json_atomic
    if getattr(original, "v1_epoch_preflight_guard", False):
        raise ValueError("Telemetry preflight publication guard is already installed")
    native = Path(request["native_reference"]).resolve(strict=True)
    expected_path = native / "results/v1.0/checkpoint_best.pt.preflight.json"

    @functools.wraps(original)
    def guarded(path, payload):
        resolved = Path(path).resolve()
        if resolved != expected_path:
            return original(path, payload)
        if request["mode"] != "native":
            raise ValueError("Nested training cannot replace native preflight calibration")
        verify_baseline_binding(request)
        _validate_calibration(native, payload)
        if resolved.exists():
            if _read(resolved) != payload:
                raise ValueError("Existing native preflight differs; no overwrite permitted")
            # Even an equivalent serialization must not change existing bytes.
            result = None
        else:
            result = original(path, payload)
        # Hash the published file, not an in-memory serialization of the probe.
        lock = verify_execution_lock(request, create=True)
        print(f"Native execution lock published | physical_batch={lock['selected_batch_size']} "
              f"| workers={lock['selected_num_workers']} | before epoch training", flush=True)
        return result

    guarded.v1_epoch_preflight_guard = True
    pipeline._write_json_atomic = guarded
    return original


class Tee:
    """Forward terminal properties so tqdm retains the original progress mode."""

    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        result = self.terminal.write(value)
        self.log.write(value)
        return result

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def invocation_logs(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    invocation = uuid.uuid4().hex
    stdout_path = output / f"invocation_{invocation}.stdout.log"
    stderr_path = output / f"invocation_{invocation}.stderr.log"
    with stdout_path.open("x", encoding="utf-8", buffering=1) as stdout_log:
        with stderr_path.open("x", encoding="utf-8", buffering=1) as stderr_log:
            previous_stdout, previous_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(previous_stdout, stdout_log), Tee(previous_stderr, stderr_log)
            try:
                yield stdout_path, stderr_path
            except BaseException:
                traceback.print_exc(file=sys.stderr)
                raise
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    sys.stdout, sys.stderr = previous_stdout, previous_stderr


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--request", required=True)
    args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if not remaining or remaining[0] != "train":
        parser.error("Epoch recording requires original train arguments")
    contract_path = Path(args.contract).resolve(strict=True)
    request = _read(Path(args.request).resolve(strict=True))
    from hiercp_v1x.sampling_runtime import (
        ENVIRONMENT_KEY, install_from_environment, load_environment_contract,
    )
    os.environ[ENVIRONMENT_KEY] = str(contract_path)
    contract = load_environment_contract()
    experiment = validate_request(request, contract_path, contract)
    source = experiment / "source/v1.0"
    _verify_contract_files(source, contract)
    if (Path(__file__).resolve() != source / "hiercp_v1x/telemetry_entry.py"
            or "hiercp_v1x/telemetry_entry.py" not in contract["source_identity"]["files"]
            or "hiercp_v1x/epoch_telemetry.py" not in contract["source_identity"]["files"]):
        raise ValueError("Telemetry entry must belong to the actual bound recording snapshot")
    with invocation_logs(request["output_dir"]) as paths:
        print(f"Foreground v1.0 epoch recording | mode={request['mode']} "
              f"| stdout={paths[0]} | stderr={paths[1]}", flush=True)
        pipeline = importlib.import_module("hiercp.pipeline")
        if Path(pipeline.__file__).resolve() != source / "hiercp/pipeline.py":
            raise ValueError("Telemetry imported another original pipeline snapshot")
        install_from_environment()
        from hiercp_v1x.sampling_entry import preserve_worker_calibration_rng
        preserve_worker_calibration_rng(pipeline)
        native_calibration = Path(request["native_reference"]) / "results/v1.0/checkpoint_best.pt.preflight.json"
        if request["mode"] == "strict_nested" or native_calibration.exists():
            verify_execution_lock(request)
        original_writer = install_preflight_publication_guard(pipeline, request)
        session = None
        previous_argv = sys.argv
        try:
            from hiercp_v1x import epoch_telemetry
            session = epoch_telemetry.install(pipeline, request["output_dir"], metadata=request)
            print(f"Epoch records | {session.record_path}", flush=True)
            sys.argv = ["hiercp.pipeline", *remaining]
            return pipeline.main()
        finally:
            sys.argv = previous_argv
            try:
                if session is not None:
                    session.close()
            finally:
                pipeline._write_json_atomic = original_writer


if __name__ == "__main__":
    main()
