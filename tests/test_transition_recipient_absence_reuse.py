"""UNIT storage-contract fixtures, never fabricated CT/neural quality evidence."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from hiercp_v22.storage import encode
from hiercp_v1x import transition_v1_local as local
from tests import test_transition_preparation_reuse as fixtures
from tools import reuse_v17_D_preparation as reuse


class RecipientAbsenceReuseUNIT(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PreparationReuseUnit()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        published = json.loads(Path(__file__).with_name("v17_aa28082_sources_UNIT.json").read_text())
        contract = local.local_contract()
        old_local = dict(contract=contract, contract_sha256=hashlib.sha256(reuse._json(contract)).hexdigest(),
                         module_sha256=reuse.PUBLISHED_LOCAL_MODULE_SHA256)
        self.old_request = copy.deepcopy(f.request)
        self.old_request["local_identity"] = old_local
        f.request = copy.deepcopy(self.old_request)
        f.request["local_identity"] = local.source_identity()
        f.identity["source"]["archive_sha256"] = contract["original_archive_sha256"]
        f.identity["sources"] = published
        f.new_identity = copy.deepcopy(f.identity)
        f.new_identity["sources"][reuse._LOCAL_MODULE] = f.request["local_identity"]["module_sha256"]
        f.new_identity["sources"][reuse._ABSENCE_MODULE] = f.request["local_identity"]["recipient_absence_adapter"]["module_sha256"]
        f.new_identity["sources"]["tools/reuse_v17_D_preparation.py"] = reuse._sha(Path(reuse.__file__))
        self.write_metadata()
        self.records = []
        self.source = dict(source_local=self.branch("source_context", 2), source_patch=torch.ones(5, 2, 2, 2))
        self.publish_source()
        for ordinal, row in enumerate(f.rows):
            record = dict(format=local.FORMAT, case_id=row["case_id"], donor_case_id=row["donor_case_id"],
                component_id=row["donor_component"], center=row["center"], scope_contract=f.request["scope_contract"],
                input_provenance=dict(observation_id=row["id"]), target_erasure=True, views_per_observation=2,
                audit={}, target_local=self.branch("target_context", ordinal+3),
                target_patch=torch.ones(5, 2, 2, 2))
            self.records.append(record)
            self.publish_record(ordinal)

    def write_metadata(self):
        f = self.fixture
        fixtures.dump(f.old / "prepare_request.json", self.old_request)
        fixtures.dump(f.old.parent / "manifest.json", f.identity)
        fixtures.dump(f.dest.parent / "manifest.json", f.new_identity)

    def branch(self, role, count):
        return dict(format="canonical-full-v22", nodes={role: dict(x=torch.ones(count, 16))},
                    counts={role: count}, edges={},
                    v1x_bounded_scope_contract=self.fixture.request["scope_contract"])

    def publish_source(self):
        f = self.fixture
        raw = encode(self.source)
        digest = hashlib.sha256(raw).hexdigest()
        f.source_relative = f.segment + "/shared_sources/" + digest + ".pt.gz"
        f.source_file = f.old / f.source_relative
        f.source_file.write_bytes(gzip.compress(raw, compresslevel=1, mtime=0))
        for row in f.stored:
            row["shared_source"] = dict(path=f.source_relative.split(f.segment+"/")[1],
                sha256=reuse._sha(f.source_file), content_sha256=digest)
            row["compressed_shared_source_bytes"] = f.source_file.stat().st_size

    def publish_record(self, ordinal):
        f = self.fixture
        record = self.records[ordinal]
        record["content_binding"] = dict(format=local.FORMAT,
            graph_sha256=local._graph_hash({**record, **self.source}))
        path = f.old / f.segment / f.stored[ordinal]["path"]
        payload = {**record, "shared_source": f.stored[ordinal]["shared_source"]}
        path.write_bytes(gzip.compress(encode(payload), compresslevel=1, mtime=0))
        f.stored[ordinal]["sha256"] = reuse._sha(path)
        f.stored[ordinal]["compressed_graph_bytes"] = path.stat().st_size

    def test_fixture_is_exact_published_176_source_inventory(self):
        f = self.fixture
        self.assertEqual(len(f.identity["sources"]), 176)
        self.assertEqual(hashlib.sha256(reuse._json(f.identity["sources"])).hexdigest(),
                         reuse.PUBLISHED_EXECUTION_SOURCE_SHA256)

    def test_read_only_compatibility_import_checks_every_completed_row(self):
        f = self.fixture
        f.receipt(0); f.receipt(1)
        before = {str(p.relative_to(f.old)): reuse._sha(p) for p in f.old.rglob("*") if p.is_file()}
        result = f.run_import()
        proof = result["recipient_absence_continuation"]
        self.assertEqual(proof["old_release_commit"], reuse.PUBLISHED_PREPARATION_COMMIT)
        self.assertEqual(proof["verified_completed_nonempty_observations"], 2)
        self.assertEqual(proof["source_context_nodes"], 4)
        self.assertEqual(proof["target_context_nodes"], 7)
        self.assertEqual(proof["record_bytes_changed"], 0)
        self.assertEqual(proof["canonical_graphs_rebuilt"], 0)
        self.assertEqual(proof["compatibility_workers"], 2)
        self.assertTrue(result["graph_deserialization"])
        self.assertFalse(result["graph_reconstruction"])
        self.assertFalse((f.dest / "index.json").exists())
        self.assertTrue(os.path.samefile(f.source_file, f.dest / f.source_relative))
        self.assertEqual(before, {str(p.relative_to(f.old)): reuse._sha(p) for p in f.old.rglob("*") if p.is_file()})
        self.assertEqual(result, f.run_import())

    def test_uncompleted_graph_is_never_deserialized_or_imported(self):
        f = self.fixture
        f.receipt(0)
        self.records[1]["target_local"]["counts"]["target_context"] = 0
        self.publish_record(1)
        result = f.run_import()
        self.assertEqual(result["imported_completed_observations"], 1)
        self.assertEqual(result["recipient_absence_continuation"]["verified_completed_nonempty_observations"], 1)
        self.assertFalse((f.dest / f.segment / f.stored[1]["path"]).exists())

    def test_unpublished_local_source_is_rejected(self):
        self.old_request["local_identity"]["module_sha256"] = "7"*64
        self.write_metadata(); self.fixture.rejected()

    def test_unpublished_execution_inventory_is_rejected(self):
        self.fixture.identity["sources"]["hiercp_v1x/transition_v1_data.py"] = "7"*64
        self.write_metadata(); self.fixture.rejected()

    def test_current_adapter_hash_tamper_is_rejected(self):
        self.fixture.request["local_identity"]["recipient_absence_adapter"]["module_sha256"] = "7"*64
        self.fixture.rejected()

    def test_current_local_identity_cannot_be_invented(self):
        self.fixture.request["local_identity"]["module_sha256"] = "7"*64
        self.fixture.rejected()

    def test_new_scope_change_is_rejected(self):
        self.fixture.request["scope_contract"] = "7"*64
        self.fixture.rejected()

    def test_new_model_graph_contract_change_is_rejected(self):
        self.fixture.request["local_identity"]["contract"]["local_layers"] = 2
        self.fixture.rejected()

    def test_unrelated_neural_execution_change_is_rejected(self):
        self.fixture.new_identity["sources"]["hiercp_v1x/transition_model.py"] = "7"*64
        self.write_metadata(); self.fixture.rejected()

    def test_absence_adapter_must_be_in_current_execution_closure(self):
        del self.fixture.new_identity["sources"][reuse._ABSENCE_MODULE]
        self.write_metadata(); self.fixture.rejected()

    def test_new_adapter_hash_in_manifest_must_match_actual_file(self):
        self.fixture.new_identity["sources"][reuse._ABSENCE_MODULE] = "7"*64
        self.write_metadata(); self.fixture.rejected()

    def test_extra_execution_module_is_rejected(self):
        self.fixture.new_identity["sources"]["hiercp_v1x/UNIT_other_geometry.py"] = "7"*64
        self.write_metadata(); self.fixture.rejected()

    def test_old_empty_target_context_cannot_be_relabelled(self):
        self.records[0]["target_local"] = self.branch("target_context", 0)
        self.publish_record(0); self.fixture.receipt(0); self.fixture.rejected()

    def test_old_empty_donor_context_cannot_be_relabelled(self):
        self.source["source_local"] = self.branch("source_context", 0)
        self.publish_source(); self.publish_record(0)
        self.fixture.receipt(0); self.fixture.rejected()

    def test_old_count_cannot_hide_empty_target_context(self):
        self.records[0]["target_local"]["nodes"]["target_context"]["x"] = torch.empty(0, 16)
        self.publish_record(0); self.fixture.receipt(0); self.fixture.rejected()

    def test_nonfinite_context_is_rejected_even_with_rebound_content_hash(self):
        self.records[0]["target_local"]["nodes"]["target_context"]["x"][0, 0] = float("nan")
        self.publish_record(0); self.fixture.receipt(0); self.fixture.rejected()

    def test_old_record_cannot_add_new_absence_marker(self):
        self.records[0]["target_local"]["transition_recipient_context_absence"] = {"UNIT_absence": True}
        self.publish_record(0); self.fixture.receipt(0); self.fixture.rejected()

    def test_old_audit_cannot_add_new_absence_marker(self):
        self.records[0]["audit"]["recipient_context_absence"] = {"UNIT_absence": True}
        self.publish_record(0); self.fixture.receipt(0); self.fixture.rejected()

    def test_record_assignment_and_binding_are_checked(self):
        f = self.fixture
        self.records[0]["donor_case_id"] = "UNIT_other_donor"
        self.publish_record(0); f.receipt(0); f.rejected()

    def test_old_record_scope_marker_is_required(self):
        del self.records[0]["target_local"]["v1x_bounded_scope_contract"]
        self.publish_record(0); self.fixture.receipt(0); self.fixture.rejected()

    def test_record_content_binding_tamper_is_rejected(self):
        f = self.fixture
        self.records[0]["target_local"]["nodes"]["target_context"]["x"].add_(1)
        path = f.old / f.segment / f.stored[0]["path"]
        path.write_bytes(gzip.compress(encode({**self.records[0], "shared_source": f.stored[0]["shared_source"]}), mtime=0))
        f.stored[0]["sha256"] = reuse._sha(path)
        f.stored[0]["compressed_graph_bytes"] = path.stat().st_size
        f.receipt(0); f.rejected()

    def test_compatibility_respects_explicit_RSS_admission(self):
        f = self.fixture
        f.receipt(0)
        with patch("psutil.Process") as process:
            process.return_value.memory_info.return_value.rss = 33 * 2**30
            with self.assertRaises(MemoryError): f.run_import()
        self.assertFalse(f.dest.exists())


if __name__ == "__main__":
    unittest.main()
