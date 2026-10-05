"""Read-only import of byte-verified D preparation receipts into a new run.

This is a storage continuation, not checkpoint migration or a graph repair.
The explicit aa28082 recipient-absence migration reads old graph payloads once
to prove their existing source/target context is nonempty. It does not rebuild,
copy, truncate, rewrite, or declare any graph/observation complete here.
The D provider still admits the imported rows and publishes the complete index.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat


FORMAT = "native_PU_actual_original_v1_full_canonical_cache_v2"
MEASUREMENT_FORMAT = "actual_original_two_view_epoch0_counts_v1"
ASSIGNMENT_KEYS = ("id", "case_id", "patient_group", "component", "center", "target",
                   "donor_case_id", "donor_component", "donor_group")
IDENTITY_KEYS = ("format", "arm", "debug", "scope", "native_inventory_sha256",
                 "native_experiment_binding", "source", "settings", "hardware",
                 "original_curriculum_binding")
ALLOWED_EXECUTION_CHANGES = frozenset(("hiercp_v1x/transition_v1_data.py", "hiercp_v1x/transition_preparation_storage.py",
    "tools/run_v17_crossed_training.py", "tools/reuse_v17_D_preparation.py"))
PUBLISHED_PREPARATION_COMMIT = "aa280829d018ed0f4426a03d59f794b724d76bb0"
# These are hashes of the actual published checkout bytes, with its explicit
# .gitattributes CRLF rules applied, independently verified against git archive.
# The source-inventory digest covers all176 execution files in that release.
PUBLISHED_EXECUTION_SOURCE_SHA256 = "f5ffde4c3bd64750094cb822bbc4d7f42d4c58bcfbbe90883ecf755aeb2f9ef8"
PUBLISHED_LOCAL_MODULE_SHA256 = "8bfe9466d23bab7f33fb11678dc3017f0a6cde1c375d48e86957198ee2a30d49"
PUBLISHED_BOUNDED_SCOPE_SHA256 = "4e52f4751e9d3e885f088bdf5639cbda649099c4db52de069902772a2223853b"
_LOCAL_MODULE = "hiercp_v1x/transition_v1_local.py"
_ABSENCE_MODULE = "hiercp_v1x/transition_v1_empty_context.py"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SEGMENT = re.compile(r"segments/[0-9a-f]{32}\Z")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate preparation JSON key: " + key)
        value[key] = item
    return value


def _load(path):
    def invalid_constant(value):
        raise ValueError("Non-finite preparation JSON value: " + value)
    value = json.loads(path.read_text(encoding="utf8"), object_pairs_hook=_unique,
                       parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Preparation metadata must be a JSON object")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _real(path, *, directory=False):
    """Reject symlinks before resolving, including any ancestor component."""
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Preparation reuse cannot follow symlinks: " + str(part))
    resolved = path.resolve(strict=True)
    mode = resolved.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError("Expected a real preparation " + ("directory" if directory else "file"))
    return resolved


def _relative(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe preparation relative path")
    path = _real(root / PurePosixPath(name))
    if not path.is_relative_to(root):
        raise ValueError("Preparation reference escapes owned root")
    return path


def _hash_text(value):
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError("Explicit lowercase SHA256 required for reused preparation")
    return value


def _stat(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _measurements(row):
    if (row.get("sampled_measurement_format") != MEASUREMENT_FORMAT
            or type(row.get("measurement_epoch")) is not int or row["measurement_epoch"] != 0):
        raise ValueError("Both genuine epoch0 sampled-view measurements required")
    for kind in ("nodes", "edges"):
        values = row.get("sampled_view_" + kind)
        if (not isinstance(values, list) or len(values) != 2
                or any(type(value) is not int or value < (1 if kind == "nodes" else 0) for value in values)
                or type(row.get("sampled_two_view_" + kind)) is not int
                or row["sampled_two_view_" + kind] != sum(values)):
            raise ValueError("Reused actual sampled-view counts are invalid")
    bounds = row.get("bounds")
    if not isinstance(bounds, dict) or any(type(bounds.get(key)) is not int or bounds[key] < 0
                                          for key in ("nodes", "edges", "bytes")):
        raise ValueError("Reused canonical bounds are invalid")


def _continuation(request, expected_request, old_sources, new_sources):
    """Admit only storage-only reuse or the one published absence extension.

    A new absence capability never retroactively changes the meaning of an old
    record. The compatibility proof below admits only already nonempty source
    AND recipient context. Exact model/scope/data/assignment settings remain
    independently checked by the caller. Unknown historical source identities
    cannot be relabelled as this published preparation.
    """
    if request == expected_request:
        return None
    old_local = request.get("local_identity", {})
    new_local = expected_request.get("local_identity", {})
    if (old_local.get("module_sha256") != PUBLISHED_LOCAL_MODULE_SHA256
            or set(old_local) != {"contract", "module_sha256", "contract_sha256"}
            or set(new_local) != set(old_local) | {"recipient_absence_adapter"}):
        raise ValueError("Recipient-absence reuse requires the exact published aa28082 local identity")
    restored = copy.deepcopy(expected_request)
    restored["local_identity"].pop("recipient_absence_adapter")
    restored["local_identity"]["module_sha256"] = old_local["module_sha256"]
    if restored != request:
        raise ValueError("Recipient-absence continuation cannot change request data/GT/donor/scope/model/settings")
    if (not isinstance(old_sources, dict) or len(old_sources) != 176
            or hashlib.sha256(_json(old_sources)).hexdigest() != PUBLISHED_EXECUTION_SOURCE_SHA256
            or old_sources.get(_LOCAL_MODULE) != PUBLISHED_LOCAL_MODULE_SHA256
            or old_sources.get("hiercp_v1x/bounded_scope.py") != PUBLISHED_BOUNDED_SCOPE_SHA256
            or _ABSENCE_MODULE in old_sources):
        raise ValueError("Recipient-absence reuse requires the exact published aa28082 execution source inventory")
    from hiercp_v1x import transition_v1_local as local
    actual = local.source_identity()
    if new_local != actual:
        raise ValueError("New recipient-absence request is not bound to the actual current adapter/local modules")
    adapter = actual["recipient_absence_adapter"]
    if (not isinstance(new_sources, dict)
            or new_sources.get(_LOCAL_MODULE) != actual["module_sha256"]
            or new_sources.get(_ABSENCE_MODULE) != adapter["module_sha256"]
            or new_sources.get("tools/reuse_v17_D_preparation.py") != _sha(Path(__file__))
            or new_sources.get("hiercp_v1x/bounded_scope.py") != PUBLISHED_BOUNDED_SCOPE_SHA256
            or set(new_sources) - set(old_sources) != {_ABSENCE_MODULE}):
        raise ValueError("New recipient-absence execution inventory is not bound to its actual modules")
    return dict(format="v17_D_published_nonempty_context_continuation_v1",
                old_release_commit=PUBLISHED_PREPARATION_COMMIT,
                old_execution_sources_sha256=PUBLISHED_EXECUTION_SOURCE_SHA256,
                old_local_module_sha256=PUBLISHED_LOCAL_MODULE_SHA256,
                new_local_module_sha256=actual["module_sha256"],
                new_recipient_absence_adapter=copy.deepcopy(adapter),
                compatibility="Old source AND recipient context nonempty; no absence marker; unchanged record bytes",
                scope_model_GT_donor_assignment_unchanged=True)


def _nonempty_original_record(root, row, expected_request):
    """One read-only compatibility proof; never construct a replacement graph."""
    import numpy as np
    import torch
    from hiercp_v22.storage import load_record
    from hiercp_v1x import transition_v1_local as local
    record = load_record(root / row["segment"], row["path"])
    if (not isinstance(record, dict) or record.get("format") != local.FORMAT
            or record.get("case_id") != row["case_id"]
            or record.get("donor_case_id") != row["donor_case_id"]
            or record.get("component_id") != row["donor_component"]
            or record.get("center") != row["center"]
            or record.get("scope_contract") != expected_request["scope_contract"]
            or record.get("input_provenance", {}).get("observation_id") != row["id"]
            or record.get("target_erasure") is not True or record.get("views_per_observation") != 2):
        raise ValueError("Reused canonical record differs from its exact original observation/scope")
    if ("recipient_context_absence" in record.get("audit", {})
            or "transition_recipient_context_absence" in record.get("target_local", {})):
        raise ValueError("Old preparation cannot contain a new recipient-absence marker")
    counts = {}
    for branch, role in (("source_local", "source_context"), ("target_local", "target_context")):
        payload = record.get(branch, {})
        node = payload.get("nodes", {}).get(role, {})
        features = node.get("x")
        count = payload.get("counts", {}).get(role)
        if (not isinstance(features, torch.Tensor) or features.device.type != "cpu"
                or features.ndim != 2 or features.shape[1] != 16 or features.shape[0] <= 0
                or type(count) is not int or count != features.shape[0]
                or payload.get("v1x_bounded_scope_contract") != expected_request["scope_contract"]):
            raise ValueError("Old preparation needs real nonempty source AND recipient context: " + row["id"])
        if not np.isfinite(features.detach().numpy()).all():
            raise ValueError("Reused original context features contain nonfinite values: " + row["id"])
        counts[role] = count
    binding = record.get("content_binding", {})
    digest = _hash_text(binding.get("graph_sha256"))
    if binding.get("format") != local.FORMAT or local._graph_hash(record) != digest:
        raise ValueError("Reused canonical record content binding differs")
    return dict(id=row["id"], graph_content_sha256=digest, **counts)


def import_preparation(source, destination, expected_request, *, expected_rows, expected_identity):
    """Import completed rows as hardlinks; leave all old files untouched.

    ``source`` accepts either an old run root or its canonical_cache directory.
    ``destination`` must be a new canonical_cache directory in an already-owned
    new run. The caller supplies the exact current preparation request, ordered
    native rows, and new run manifest. Only admitted execution-source hashes may differ;
    neural recipe, source snapshot, data, hardware and settings stay identical.
    A failure preserves any newly linked files for inspection and never changes
    old files. A complete import is identified by reuse_receipt.json.
    """
    root = _real(source, directory=True)
    if not (root / "prepare_request.json").is_file():
        root = _real(root / "canonical_cache", directory=True)
    dest = Path(destination).absolute()
    parent = _real(dest.parent, directory=True)
    dest = parent / dest.name
    if dest.is_symlink():
        raise ValueError("Preparation import destination cannot be a symlink")
    if dest.is_relative_to(root) or root.is_relative_to(dest) or parent == root.parent:
        raise ValueError("Old and new preparation runs must be separate, non-nested roots")
    if _load(_real(parent / "manifest.json")) != expected_identity:
        raise ValueError("Destination is not owned by the exact new D run manifest")
    new_lock = parent / ".experiment.lock"
    if new_lock.exists() or new_lock.is_symlink():
        owner = _load(_real(new_lock / "owner.json"))
        if owner.get("pid") != os.getpid() or owner.get("host") != socket.gethostname():
            raise ValueError("Destination run is locked by another process; no lock removal")

    request_path = _real(root / "prepare_request.json")
    lock = root.parent / ".experiment.lock"
    if lock.exists() or lock.is_symlink():
        raise ValueError("Old D run is active/locked; interrupt its own foreground process before reuse")
    request = _load(request_path)
    if request.get("format") != FORMAT:
        raise ValueError("Old preparation does not match the actual original canonical format")
    if type(request.get("debug")) is not bool or type(request.get("prepared_observations")) is not int:
        raise ValueError("Explicit preparation scope/count required")
    for key in ("input_inventory_sha256", "assignment_sha256", "prepared_assignment_sha256", "scope_contract"):
        _hash_text(request.get(key))
    identity = request.get("local_identity", {})
    contract = identity.get("contract", {})
    if (contract.get("format") != "native_PU_actual_original_v1_local_two_view_10mm_v1"
            or contract.get("ROI_margin_mm") != 10
            or request.get("base", {}).get("graph", {}).get("adaptive_roi_margin_mm") != 10
            or hashlib.sha256(_json(contract)).hexdigest() != identity.get("contract_sha256")):
        raise ValueError("Actual original two-view ten-mm D contract required")
    _hash_text(identity.get("module_sha256"))
    archive = _hash_text(contract.get("original_archive_sha256"))

    manifest_path = _real(root.parent / "manifest.json")
    manifest = _load(manifest_path)
    if (manifest.get("format") != "v17_crossed_training_identity_v1" or manifest.get("arm") != "D"
            or manifest.get("debug") is not request["debug"]
            or manifest.get("scope") != request["scope_contract"]
            or manifest.get("native_inventory_sha256") != request["input_inventory_sha256"]
            or manifest.get("source", {}).get("archive_sha256") != archive
            or any(key not in manifest or key not in expected_identity
                   or manifest[key] != expected_identity[key] for key in IDENTITY_KEYS)):
        raise ValueError("Old D run recipe/data/source/hardware/settings differs; no automatic migration")
    old_sources, new_sources = manifest.get("sources"), expected_identity.get("sources")
    if not isinstance(old_sources, dict) or not isinstance(new_sources, dict) or not old_sources or not new_sources:
        raise ValueError("Both exact execution-source inventories are required")
    continuation = _continuation(request, expected_request, old_sources, new_sources)
    allowed_changes = ALLOWED_EXECUTION_CHANGES
    if continuation is not None:
        allowed_changes |= {_LOCAL_MODULE, _ABSENCE_MODULE}
    for name in old_sources.keys() | new_sources.keys():
        if name not in allowed_changes and old_sources.get(name) != new_sources.get(name):
            raise ValueError("Execution change is outside the storage/preparation continuation: " + name)
        if name in old_sources:
            _hash_text(old_sources[name])
        if name in new_sources:
            _hash_text(new_sources[name])

    rows = list(expected_rows)
    if not rows or len(rows) != request["prepared_observations"]:
        raise ValueError("Complete expected native preparation rows required")
    assignments = [{key: copy.deepcopy(row[key]) for key in ASSIGNMENT_KEYS} for row in rows]
    if hashlib.sha256(_json(assignments)).hexdigest() != request["prepared_assignment_sha256"]:
        raise ValueError("Expected native preparation assignment digest differs")
    originals = {row["id"]: (ordinal, row) for ordinal, row in enumerate(rows)}
    if len(originals) != len(rows) or any(not isinstance(key, str) or not key for key in originals):
        raise ValueError("Unique nonempty native observation IDs required")
    if not request["debug"] and len(rows) != 14102:
        raise ValueError("Production reuse must preserve all14102 native observations")
    ledger = _real(root / "completed", directory=True)
    receipts = sorted(ledger.glob("*.json"))
    files, imported = {}, []
    anchors = {request_path: _stat(request_path), manifest_path: _stat(manifest_path)}
    hashes = {request_path: _sha(request_path), manifest_path: _sha(manifest_path)}
    for receipt in receipts:
        receipt = _real(receipt)
        if re.fullmatch(r"[0-9]{6}\.json", receipt.name) is None:
            raise ValueError("Reused receipt needs the original native ordinal")
        row = _load(receipt)
        original = originals.get(row.get("id"))
        if original is None or receipt.name != f"{original[0]:06d}.json":
            raise ValueError("Reused observation ID/receipt ordinal differs")
        if any(row.get(key) != original[1][key] for key in ASSIGNMENT_KEYS):
            raise ValueError("Reused observation GT/center/donor assignment differs")
        segment = row.get("segment")
        if not isinstance(segment, str) or _SEGMENT.fullmatch(segment) is None:
            raise ValueError("Owned original D segment required")
        if row.get("path") != f"graphs/{row['case_id']}/{original[0]:06d}.pt.gz":
            raise ValueError("Reused graph is not the original observation ordinal")
        _measurements(row)
        reference = row.get("shared_source")
        if not isinstance(reference, dict):
            raise ValueError("Exact shared donor source reference required")
        content_hash = _hash_text(reference.get("content_sha256"))
        if reference.get("path") != f"shared_sources/{content_hash}.pt.gz":
            raise ValueError("Shared donor content path differs")
        for metadata, size_key in ((row, "compressed_graph_bytes"),
                                   (reference, "compressed_shared_source_bytes")):
            relative = segment + "/" + metadata["path"]
            path = _relative(root, relative)
            digest = _hash_text(metadata.get("sha256"))
            size = row.get(size_key)
            if type(size) is not int or size <= 0 or path.stat().st_size != size:
                raise ValueError("Reused compressed-file extent differs")
            if relative in files:
                if files[relative][1] != digest:
                    raise ValueError("One reused file has contradictory byte identities")
            else:
                snapshot = _stat(path)
                if _sha(path) != digest or _stat(path) != snapshot:
                    raise ValueError("Reused graph/shared-source SHA256 changed")
                files[relative] = (path, digest, snapshot)
        anchors[receipt], hashes[receipt] = _stat(receipt), _sha(receipt)
        imported.append((receipt.name, row))
    if continuation is not None:
        # Only this narrow compatibility migration reads graph payloads. Each
        # completed row is read once, in the same explicit admitted worker pool;
        # no original raw CT/canonical graph is reconstructed. The compressed
        # files remain read-only and keep their original content/byte hashes.
        from concurrent.futures import ThreadPoolExecutor
        import psutil
        workers = expected_identity["settings"].get("workers")
        rss_gib = expected_identity["settings"].get("rss_gib")
        if type(workers) is not int or workers < 2 or not isinstance(rss_gib, (int, float)) or rss_gib <= 0:
            raise ValueError("Explicit parallel/RSS admission required for nonempty context compatibility proof")
        print(f"D read-only reuse compatibility | {len(imported)} completed observations | "
              f"workers={workers} | unchanged10mm/source/target bytes", flush=True)
        def prove(value):
            if psutil.Process().memory_info().rss > rss_gib * 2**30:
                raise MemoryError("Recipient-absence compatibility proof exceeds admitted process RSS")
            proof = _nonempty_original_record(root, value[1], expected_request)
            if psutil.Process().memory_info().rss > rss_gib * 2**30:
                raise MemoryError("Recipient-absence compatibility proof exceeds admitted process RSS")
            return proof
        with ThreadPoolExecutor(max_workers=workers) as pool:
            proofs = []
            for proof in pool.map(prove, imported):
                proofs.append(proof)
                if len(proofs) % 64 == 0 or len(proofs) == len(imported):
                    print(f"D verified original nonempty context | {len(proofs)}/{len(imported)}", flush=True)
        continuation.update(verified_completed_nonempty_observations=len(proofs),
            compatibility_proof_sha256=hashlib.sha256(_json(proofs)).hexdigest(),
            compatibility_workers=workers, compatibility_rss_gib=rss_gib,
            source_context_nodes=sum(value["source_context"] for value in proofs),
            target_context_nodes=sum(value["target_context"] for value in proofs),
            record_bytes_changed=0, canonical_graphs_rebuilt=0)
    if (lock.exists() or lock.is_symlink() or sorted(ledger.glob("*.json")) != receipts
            or any(_stat(path) != anchors[path] or _sha(path) != hashes[path] for path in anchors)):
        raise ValueError("Old preparation is changing; finish interrupt before importing")

    receipt = dict(format="v17_D_verified_preparation_import_v1", source=str(root), destination=str(dest),
                   input_inventory_sha256=request["input_inventory_sha256"],
                   source_request_sha256=hashes[request_path], source_manifest_sha256=hashes[manifest_path],
                   expected_observations=len(rows), imported_completed_observations=len(imported),
                   imported_file_count=len(files), referenced_compressed_bytes=sum(x[2][2] for x in files.values()),
                   graph_bytes_copied=0, graph_deserialization=continuation is not None, graph_reconstruction=False,
                   existing_outputs_modified=False, uncommitted_rows_imported=False,
                   training_or_checkpoint_imported=False, all_rows_complete_here=False,
                   pending_observations=len(rows)-len(imported), publication_requires_provider_preflight=True,
                   imported_native_ordinals=[int(name[:6]) for name, _ in imported])
    if continuation is not None:
        receipt["recipient_absence_continuation"] = continuation
    if dest.exists():
        _real(dest, directory=True)
        if not (dest / "reuse_receipt.json").is_file():
            raise ValueError("Existing destination lacks its complete exact import receipt; no overwrite")
        if (_load(_real(dest / "reuse_receipt.json")) != receipt
                or _load(_real(dest / "prepare_request.json")) != expected_request):
            raise ValueError("Existing preparation import belongs to another exact source/request")
        for name, row in imported:
            if _load(_real(dest / "completed" / name)) != row:
                raise ValueError("Existing imported completion receipt changed")
        for relative, (path, digest, _) in files.items():
            target = _relative(dest, relative)
            if not os.path.samefile(path, target) or _sha(target) != digest:
                raise ValueError("Existing imported graph is not its verified original hardlink")
        # The provider may have appended newly completed observations meanwhile.
        # Their independent request/GT/hash checks still run in preflight. Here
        # only the exact imported subset is re-admitted without replacing files.
        return receipt

    # Validate everything first. New files are confined to this new owned root.
    # Graph bytes stay on the same filesystem. No expensive graph reconstruction
    # or serialization occurs, and an unsupported hardlink is an explicit error.
    if root.stat().st_dev != parent.stat().st_dev:
        raise ValueError("Reuse needs the same filesystem for hardlinks; no graph copy fallback")
    dest.mkdir(exist_ok=False)
    (dest / "completed").mkdir()
    _write(dest / "prepare_request.json", expected_request)
    for relative, (path, digest, snapshot) in sorted(files.items()):
        if _stat(path) != snapshot:
            raise ValueError("Old preparation file changed before hardlink import")
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        if _sha(target) != digest or _stat(path) != snapshot:
            raise ValueError("Imported hardlink bytes changed during import")
    for name, row in imported:
        _write(dest / "completed" / name, row)
    if (lock.exists() or lock.is_symlink() or sorted(ledger.glob("*.json")) != receipts
            or any(_stat(path) != anchors[path] or _sha(path) != hashes[path] for path in anchors)):
        raise ValueError("Old preparation changed during import; new partial files preserved for inspection")
    _write(dest / "reuse_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-request", type=Path, required=True)
    parser.add_argument("--expected-native-rows", type=Path, required=True)
    parser.add_argument("--expected-run-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = _load(_real(args.expected_native_rows))["records"]
    result = import_preparation(args.source, args.destination, _load(_real(args.expected_request)),
                                expected_rows=rows, expected_identity=_load(_real(args.expected_run_manifest)))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
