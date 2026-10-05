"""Storage-only UNIT metadata; never a fabricated neural/CT result."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import reuse_v17_D_preparation as reuse


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


class PreparationReuseUnit(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="UNIT_D_reuse_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old = self.root / "old" / "canonical_cache"
        self.dest = self.root / "new" / "canonical_cache"
        self.dest.parent.mkdir()
        self.rows = [dict(id=f"UNIT_CASE:{index}", case_id="UNIT_CASE", patient_group="case:UNIT_CASE",
            component=index+1, center=[4+index, 5, 6], target=1-index, donor_case_id="UNIT_DONOR",
            donor_component=7, donor_group="case:UNIT_DONOR") for index in range(2)]
        contract = dict(format="native_PU_actual_original_v1_local_two_view_10mm_v1", ROI_margin_mm=10,
                        original_archive_sha256="a"*64)
        self.request = dict(format=reuse.FORMAT, input_inventory="UNIT_NATIVE_INVENTORY_ONLY",
            input_inventory_sha256="b"*64, assignment_sha256="c"*64,
            prepared_assignment_sha256=hashlib.sha256(reuse._json(self.rows)).hexdigest(),
            prepared_observations=2, debug=True, scope_contract="d"*64,
            local_identity=dict(contract=contract, contract_sha256=hashlib.sha256(reuse._json(contract)).hexdigest(),
                                module_sha256="e"*64),
            base=dict(graph=dict(adaptive_roi_margin_mm=10)))
        self.identity = dict(format="v17_crossed_training_identity_v1", arm="D", debug=True,
            scope="d"*64, native_inventory_sha256="b"*64,
            native_experiment_binding=dict(debug=True, inventory_sha256="b"*64),
            source=dict(archive_sha256="a"*64, verified_files={"UNIT_original.py": "0"*64}),
            sources={"hiercp_v1x/transition_v1_data.py": "1"*64, "hiercp_v1x/transition_model.py": "2"*64},
            settings=dict(workers=2, physical_batch_candidates=[2], cuda_gib=12, rss_gib=32),
            hardware=dict(name="UNIT_NO_GPU_EXECUTION", total_memory=12),
            original_curriculum_binding=dict(UNIT_preserved=True))
        self.new_identity = copy.deepcopy(self.identity)
        self.new_identity["sources"]["hiercp_v1x/transition_v1_data.py"] = "3"*64
        self.new_identity["sources"]["tools/reuse_v17_D_preparation.py"] = "4"*64
        dump(self.dest.parent / "manifest.json", self.new_identity)
        dump(self.old / "prepare_request.json", self.request)
        dump(self.old.parent / "manifest.json", self.identity)
        (self.old / "completed").mkdir()
        self.segment = "segments/" + "5"*32
        self.source_relative = self.segment + "/shared_sources/" + "6"*64 + ".pt.gz"
        self.source_file = self.old / self.source_relative
        self.source_file.parent.mkdir(parents=True)
        self.source_file.write_bytes(b"UNIT_ONLY_STORAGE_BYTES_NO_TORCH_PAYLOAD")
        self.stored = []
        for ordinal, row in enumerate(self.rows):
            path = f"graphs/UNIT_CASE/{ordinal:06d}.pt.gz"
            graph = self.old / self.segment / path
            graph.parent.mkdir(parents=True, exist_ok=True)
            graph.write_bytes(f"UNIT_ONLY_GRAPH_STORAGE_{ordinal}".encode())
            stored = dict(row, segment=self.segment, path=path, sha256=reuse._sha(graph),
                shared_source=dict(path=self.source_relative.split(self.segment+"/")[1],
                                   sha256=reuse._sha(self.source_file), content_sha256="6"*64),
                bounds=dict(nodes=10, edges=20, bytes=100),
                sampled_measurement_format=reuse.MEASUREMENT_FORMAT, measurement_epoch=0,
                sampled_view_nodes=[4, 5], sampled_view_edges=[6, 7],
                sampled_two_view_nodes=9, sampled_two_view_edges=13,
                compressed_graph_bytes=graph.stat().st_size,
                compressed_shared_source_bytes=self.source_file.stat().st_size)
            self.stored.append(stored)

    def receipt(self, index):
        dump(self.old / "completed" / f"{index:06d}.json", self.stored[index])

    def run_import(self, **changes):
        arguments = dict(source=self.old, destination=self.dest, expected_request=self.request,
                         expected_rows=self.rows, expected_identity=self.new_identity)
        arguments.update(changes)
        return reuse.import_preparation(**arguments)

    def rejected(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_import()
        self.assertFalse(self.dest.exists(), "Invalid metadata must fail before creating destination")

    def test_partial_import_has_exact_links_and_no_uncommitted_rows(self):
        self.receipt(0)
        old_files = {str(path.relative_to(self.old)): reuse._sha(path)
                     for path in self.old.rglob("*") if path.is_file()}
        result = self.run_import()
        self.assertEqual(result["imported_completed_observations"], 1)
        self.assertEqual(result["pending_observations"], 1)
        self.assertEqual(result["imported_native_ordinals"], [0])
        self.assertEqual(result["graph_bytes_copied"], 0)
        self.assertFalse(result["graph_deserialization"])
        self.assertFalse((self.dest / self.segment / self.stored[1]["path"]).exists())
        self.assertFalse((self.dest / "index.json").exists())
        graph_relative = self.segment + "/" + self.stored[0]["path"]
        self.assertTrue(os.path.samefile(self.old / graph_relative, self.dest / graph_relative))
        self.assertTrue(os.path.samefile(self.source_file, self.dest / self.source_relative))
        self.assertEqual(old_files, {str(path.relative_to(self.old)): reuse._sha(path)
                        for path in self.old.rglob("*") if path.is_file()})

    def test_complete_import_leaves_publication_to_provider(self):
        self.receipt(0); self.receipt(1)
        result = self.run_import(source=self.old.parent)
        self.assertEqual(result["imported_completed_observations"], 2)
        self.assertEqual(result["pending_observations"], 0)
        self.assertEqual(result["imported_file_count"], 3)
        self.assertTrue(result["publication_requires_provider_preflight"])
        self.assertFalse(result["all_rows_complete_here"])
        self.assertFalse((self.dest / "index.json").exists())

    def test_empty_ledger_imports_no_uncommitted_files(self):
        result = self.run_import()
        self.assertEqual(result["imported_file_count"], 0)
        self.assertEqual(result["pending_observations"], 2)

    def test_active_old_run_is_rejected(self):
        self.receipt(0)
        (self.old.parent / ".experiment.lock").mkdir()
        self.rejected()

    def test_changed_gt_is_rejected(self):
        self.stored[0]["target"] = 0
        self.receipt(0); self.rejected()

    def test_missing_graph_is_rejected(self):
        self.receipt(0)
        (self.old / self.segment / self.stored[0]["path"]).unlink()
        self.rejected()

    def test_changed_graph_bytes_is_rejected(self):
        self.receipt(0)
        path = self.old / self.segment / self.stored[0]["path"]
        path.write_bytes(b"X"*path.stat().st_size)
        self.rejected()

    def test_missing_hash_is_rejected(self):
        del self.stored[0]["sha256"]
        self.receipt(0); self.rejected()

    def test_extent_mismatch_is_rejected(self):
        self.stored[0]["compressed_graph_bytes"] += 1
        self.receipt(0); self.rejected()

    def test_wrong_ordinal_is_rejected(self):
        dump(self.old / "completed" / "000099.json", self.stored[0])
        self.rejected()

    def test_measurement_inconsistent_is_rejected(self):
        self.stored[0]["sampled_two_view_nodes"] = 8
        self.receipt(0); self.rejected()

    def test_bad_segment_paths_are_rejected(self):
        for value in ("../outside", "/absolute", "C:/drive", "segments\\escape", "segments/../escape"):
            with self.subTest(value=value):
                self.stored[0]["segment"] = value
                self.receipt(0); self.rejected()

    def test_existing_destination_is_rejected_without_overwrite(self):
        self.dest.mkdir()
        marker = self.dest / "UNIT_preserved"
        marker.write_bytes(b"UNIT_original")
        with self.assertRaises(ValueError): self.run_import()
        self.assertEqual(marker.read_bytes(), b"UNIT_original")

    def test_same_import_is_idempotent_after_provider_continuation(self):
        self.receipt(0)
        first = self.run_import()
        # An exact new observation can be admitted by provider preflight later.
        # Import continuation must not replace/delete any new provider files.
        marker = self.dest / "UNIT_provider_addition"
        marker.write_bytes(b"UNIT_new_owned_file")
        dump(self.dest / "completed" / "000001.json", self.stored[1])
        before = {str(path.relative_to(self.dest)): reuse._sha(path)
                  for path in self.dest.rglob("*") if path.is_file()}
        self.assertEqual(first, self.run_import())
        self.assertEqual(before, {str(path.relative_to(self.dest)): reuse._sha(path)
                  for path in self.dest.rglob("*") if path.is_file()})

    def test_reused_import_receipt_tamper_is_rejected(self):
        self.receipt(0); self.run_import()
        receipt = json.loads((self.dest / "reuse_receipt.json").read_text())
        receipt["imported_completed_observations"] = 2
        dump(self.dest / "reuse_receipt.json", receipt)
        with self.assertRaises(ValueError): self.run_import()

    def test_reused_completion_receipt_tamper_is_rejected(self):
        self.receipt(0); self.run_import()
        row = copy.deepcopy(self.stored[0]); row["target"] = 0
        dump(self.dest / "completed" / "000000.json", row)
        with self.assertRaises(ValueError): self.run_import()

    def test_nested_destination_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_import(destination=self.old / "child")
        self.assertFalse((self.old / "child").exists())

    def test_settings_hardware_snapshot_change_is_rejected(self):
        for key in ("settings", "hardware", "source", "native_experiment_binding"):
            with self.subTest(key=key):
                identity = copy.deepcopy(self.new_identity)
                identity[key]["UNIT_changed"] = True
                dump(self.dest.parent / "manifest.json", identity)
                with self.assertRaises(ValueError): self.run_import(expected_identity=identity)
                self.assertFalse(self.dest.exists())

    def test_neural_source_change_is_rejected(self):
        self.new_identity["sources"]["hiercp_v1x/transition_model.py"] = "7"*64
        dump(self.dest.parent / "manifest.json", self.new_identity)
        self.rejected()

    def test_request_change_is_rejected(self):
        request = copy.deepcopy(self.request)
        request["base"]["graph"]["adaptive_roi_margin_mm"] = 20
        with self.assertRaises(ValueError): self.run_import(expected_request=request)
        self.assertFalse(self.dest.exists())

    def test_row_population_change_is_rejected(self):
        with self.assertRaises(ValueError): self.run_import(expected_rows=self.rows[:1])
        self.assertFalse(self.dest.exists())

    def test_duplicate_json_key_is_rejected(self):
        path = self.old / "completed" / "000000.json"
        path.write_text('{"id":"one","id":"two"}', encoding="utf8")
        self.rejected()

    def test_shared_file_symlink_is_rejected(self):
        self.receipt(0)
        target = self.root / "UNIT_target.pt.gz"
        target.write_bytes(self.source_file.read_bytes())
        self.source_file.unlink()
        try:
            self.source_file.symlink_to(target)
        except OSError as error:
            self.skipTest("OS does not grant symlink creation: " + str(error))
        self.rejected()

    def test_symlink_guard_rejects_reference_before_resolution(self):
        self.receipt(0)
        original = Path.is_symlink
        def symlink(path):
            return path == self.source_file or original(path)
        with patch.object(Path, "is_symlink", symlink):
            self.rejected()

    def test_destination_manifest_must_be_owned(self):
        dump(self.dest.parent / "manifest.json", {"UNIT_wrong_destination": True})
        self.rejected()

    def test_other_destination_process_lock_is_rejected(self):
        dump(self.dest.parent / ".experiment.lock" / "owner.json",
             dict(pid=os.getpid()+1, host=socket.gethostname(), token="UNIT_only"))
        self.rejected()

    def test_calling_process_destination_lock_is_admitted(self):
        dump(self.dest.parent / ".experiment.lock" / "owner.json",
             dict(pid=os.getpid(), host=socket.gethostname(), token="UNIT_only"))
        result = self.run_import()
        self.assertEqual(result["pending_observations"], 2)


if __name__ == "__main__":
    unittest.main()
