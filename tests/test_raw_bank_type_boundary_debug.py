"""DEBUG storage/type-boundary tests, not native preprocessing or training."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from custom_trainers.onlinecp_raw_bank import (
    ENTRY_STORAGE, PASTE_CONTRACT, SOURCE_MAPPING_FORMAT, save_candidate, save_case,
)
from tools import online_cp_benchmark as online


class RawBankTypeBoundaryDebug(unittest.TestCase):
    def _fixture(self, root: Path, *, raw_entry: bool, raw_index: bool):
        """Make fully hashed audit inputs; no fabricated model output is used."""
        case_id = "DEBUG_case"
        raw_centers = np.asarray([[4, 5, 6], [9, 10, 11]], dtype=np.int32)
        native_centers = raw_centers[:, ::-1].copy()
        # Deterministic values exercise the bank schema, not GNN inference.
        scores = np.asarray([0.2, 0.8], dtype=np.float32)
        source_mask = np.zeros((3, 3, 3), dtype=bool)
        source_mask[1, 1, 1] = True
        source_ct = np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3)
        payload = {
            "candidate_raw_centers": raw_centers,
            "candidate_centers": native_centers,
            "scores": scores,
            "source_component": np.asarray([1], dtype=np.int64),
            "source_diameter_mm": np.asarray([2.0], dtype=np.float32),
        }
        if raw_entry:
            # This is the complete storage-auditor fixture, intentionally not
            # an engine execution fixture. Native equivalence is tested apart.
            case = {
                "metadata": {
                    "case_id": case_id, "preprocessed_shape": [16, 16, 16],
                    "cropped_shape": [16, 16, 16], "transpose_forward": [0, 1, 2],
                    "crop_bbox": [[0, 16], [0, 16], [0, 16]],
                },
                "baseline_seg": np.zeros((1, 16, 16, 16), dtype=np.int16),
            }
            case_reference = "raw_cases/DEBUG_case.json"
            case_sha = save_case(root, case_reference, case)
            references, digests = [], []
            for number, (raw_center, native_center) in enumerate(zip(raw_centers, native_centers)):
                reference = f"raw_candidates/DEBUG_case/{number:04d}.json"
                candidate = {
                    "case_id": case_id, "source_component": 1,
                    "case_reference_sha256": case_sha,
                    "raw_target_center": raw_center.tolist(),
                    "output_bbox": np.column_stack((native_center - 1, native_center + 2)),
                    "source_mask": source_mask, "source_ct": source_ct,
                    "pasted_support": source_mask.copy(),
                    "seg_patch": (source_mask.astype(np.int16) * 2)[None],
                }
                digests.append(save_candidate(root, reference, candidate))
                references.append(reference)
            payload.update(
                paste_contract=np.asarray([PASTE_CONTRACT]), case_id=np.asarray([case_id]),
                raw_case_reference=np.asarray([case_reference]),
                raw_case_reference_sha256=np.asarray([case_sha]),
                candidate_payloads=np.asarray(references),
                candidate_payload_sha256=np.asarray(digests),
            )
        else:
            payload.update(source_data=source_ct, source_mask=source_mask.astype(np.uint8),
                           anchor_offset=np.asarray([1, 1, 1], dtype=np.int32))
        entries = root / "entries"
        entries.mkdir()
        relative = "entries/DEBUG_case__component_001.npz"
        np.savez(root / relative, **payload)
        row = {
            "case_id": case_id, "source_component": 1, "status": "ok", "candidate_count": 2,
            "entry": relative, "entry_sha256": online.file_sha256(root / relative),
            "candidate_pool_sha256": online.candidate_pool_hash(raw_centers, native_centers, scores),
        }
        inventory = {case_id: [1]}
        index = {
            "format": online.BANK_FORMAT,
            "source_mapping_format": SOURCE_MAPPING_FORMAT if raw_index else online.SOURCE_MAPPING_FORMAT,
            "entries_by_case": {case_id: [relative]}, "eligible_sources_by_case": inventory,
            "no_eligible_cases": [], "eligible_cases": 1, "source_entries": 1,
            "total_candidates": 2, "candidate_count": 2,
        }
        if raw_index:
            index.update(paste_contract=PASTE_CONTRACT, entry_storage=ENTRY_STORAGE)
        return index, [row], inventory

    def test_completed_legacy_bank_accepts_legacy_entries(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_legacy_bank_type_") as directory:
            root = Path(directory)
            index, rows, inventory = self._fixture(root, raw_entry=False, raw_index=False)
            online._audit_completed_bank(root, index, rows, inventory, 2)

    def test_completed_raw_bank_accepts_raw_entries(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_bank_type_") as directory:
            root = Path(directory)
            index, rows, inventory = self._fixture(root, raw_entry=True, raw_index=True)
            online._audit_completed_bank(root, index, rows, inventory, 2)

    def test_completed_raw_bank_rejects_legacy_entries(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_declared_legacy_entry_") as directory:
            root = Path(directory)
            index, rows, inventory = self._fixture(root, raw_entry=False, raw_index=True)
            with self.assertRaisesRegex(online.OnlineBenchmarkError, "Raw-target bank declaration"):
                online._audit_completed_bank(root, index, rows, inventory, 2)

    def test_completed_legacy_bank_rejects_raw_entries(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_legacy_declared_raw_entry_") as directory:
            root = Path(directory)
            index, rows, inventory = self._fixture(root, raw_entry=True, raw_index=False)
            with self.assertRaisesRegex(online.OnlineBenchmarkError, "Legacy bank declaration"):
                online._audit_completed_bank(root, index, rows, inventory, 2)


if __name__ == "__main__":
    unittest.main()
