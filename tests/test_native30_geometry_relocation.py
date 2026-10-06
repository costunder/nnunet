"""UNIT cache relocation contracts only; no CT/model/accuracy evidence.

Compose the existing admission fixture without inheriting or replaying its tests.
Only fixture-owned temporary files are copied or changed. Production source and
existing experiment payloads are never modified by this suite.
"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from hiercp_v1x import native30_data_contract as data_contract
from hiercp_v1x import native30_geometry as geometry_module
from tests import test_native30_geometry as admission_fixture


class Native30GeometryRelocationUnit(unittest.TestCase):
    def setUp(self):
        self.fixture = admission_fixture.NativeGeometryAdmissionUnit(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        # The original admission fixture uses flat UNIT module paths. Give this
        # composed fixture the actual reviewed original hiercp/common.py layout.
        for module in self.fixture.modules.values():
            previous = Path(module.__file__)
            destination = self.fixture.source / "hiercp" / previous.name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(previous.read_bytes())
            module.__file__ = str(destination)
        self.producer_contract_root = self.fixture.root / "producer_contract"
        actual_receipt = data_contract.source_receipt()
        self.receipt = self._copy_contract(actual_receipt, self.producer_contract_root)
        self.receipt_patch = patch.object(data_contract, "source_receipt",
            side_effect=lambda: copy.deepcopy(self.receipt))
        self.receipt_patch.start()
        self.addCleanup(self.receipt_patch.stop)
        self.published = self.fixture.published_unit_cache()
        self.before = self.fixture.cache_snapshot(self.published.output)

    def _copy_contract(self, receipt, root):
        """Copy actual data-operator bytes and retain their existing AST proof."""
        result = copy.deepcopy(receipt)
        result["source_root"] = str(root)
        for name, evidence in result["source_files"].items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(evidence["path"], target)
            evidence.update(path=str(target), sha256=geometry_module._sha(target))
        dependency = result["actual_dependencies"]["l0_regions.donor_learning.validate_rows"]
        target = root / "l0_regions" / "donor_learning.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dependency["path"], target)
        dependency.update(path=str(target), sha256=geometry_module._sha(target))
        common = Path(self.fixture.modules["common"].__file__)
        result["actual_dependencies"]["hiercp.common"].update(
            path=str(common), sha256=geometry_module._sha(common))
        return result

    def _relocate(self):
        consumer_source = self.fixture.root / "consumer_original"
        for module in self.fixture.modules.values():
            previous = Path(module.__file__)
            target = consumer_source / "hiercp" / previous.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(previous.read_bytes())
            module.__file__ = str(target)
        self.receipt = self._copy_contract(self.receipt,
            self.fixture.root / "consumer_contract")
        return consumer_source

    def _reopen(self, source=None, *, config=None, inventory=None, read_only=True):
        return geometry_module.Native30Geometry(
            inventory or self.fixture.input,
            self.published.config if config is None else config,
            self.fixture.source if source is None else source,
            workers=2, resident_bytes=1024**3, rss_bytes=8*1024**3,
            output=self.published.output, debug=self.published.debug,
            debug_case_ids=self.published.case_ids, read_only=read_only)

    def _no_preparation_io(self):
        """SHA reads remain mandatory; forbid decoding, tensors, builds/writes."""
        stack = ExitStack()
        forbidden = AssertionError("UNIT: cache relocation must not rebuild, load tensors, or write")
        for target in ("pathlib.Path.mkdir", "torch.load",
                       "hiercp_v1x.native30_geometry._write_new",
                       "hiercp_v1x.native30_geometry._save_new"):
            stack.enter_context(patch(target, side_effect=forbidden))
        for name in ("_records_for", "_preload", "_sample", "_build", "_persist"):
            stack.enter_context(patch.object(geometry_module.Native30Geometry,
                name, side_effect=forbidden))
        return stack

    def _assert_cache_preserved(self):
        self.assertEqual(self.fixture.cache_snapshot(self.published.output), self.before)

    def test_exact_request_remains_read_only_without_build_load_or_metadata_change(self):
        with self._no_preparation_io():
            reused = self._reopen()
            receipt = reused.prepare()
        proof = receipt["reuse_provenance"]
        self.assertEqual(proof["request_compatibility"]["mode"], "exact")
        self.assertEqual(reused.request_sha256, self.published.request_sha256)
        self.assertEqual(reused.consumer_request_sha256, self.published.request_sha256)
        self.assertEqual(proof["request_compatibility"]["changed_location_fields"], [])
        self.assertEqual(receipt["geometry_builds"], 0)
        self.assertEqual(receipt["geometry_loads"], 0)
        self._assert_cache_preserved()

    def test_same_bytes_original_and_contract_worktree_relocation_retains_producer_hash(self):
        source = self._relocate()
        with self._no_preparation_io():
            reused = self._reopen(source)
            receipt = reused.prepare()
        proof = receipt["reuse_provenance"]
        compatibility = proof["request_compatibility"]
        self.assertEqual(compatibility["mode"], "data_contract_source_relocation")
        expected = {"native_data_contract_receipt.source_root",
            "native_data_contract_receipt.source_files.l0_regions/donor_data.py.path",
            "native_data_contract_receipt.source_files.hiercp_v22/data.py.path",
            "native_data_contract_receipt.actual_dependencies.hiercp.common.path",
            "native_data_contract_receipt.actual_dependencies.l0_regions.donor_learning.validate_rows.path"}
        self.assertEqual({item["field"] for item in compatibility["changed_location_fields"]}, expected)
        self.assertEqual(reused.request_sha256, self.published.request_sha256)
        self.assertEqual(proof["request_sha256"], self.published.request_sha256)
        self.assertEqual(reused.consumer_request_sha256, geometry_module._json_sha(reused.request))
        self.assertNotEqual(reused.consumer_request_sha256, reused.request_sha256)
        self.assertEqual(proof["consumer_request_sha256"], reused.consumer_request_sha256)
        self.assertEqual(proof["validated_observations"], len(self.published.rows))
        self.assertTrue(proof["all_payload_sha256_verified"])
        self.assertFalse(proof["existing_files_written"])
        self.assertEqual(receipt["raw_cases_decoded"], [])
        self.assertEqual(receipt["geometry_builds"], 0)
        self.assertEqual(receipt["geometry_loads"], 0)
        self._assert_cache_preserved()

    def test_relocation_is_read_only_only_and_mutable_reopen_stays_exact(self):
        source = self._relocate()
        with self.assertRaisesRegex(ValueError, "request differs"):
            self._reopen(source, read_only=False)
        self._assert_cache_preserved()

    def test_changed_operator_AST_dependency_bytes_and_policy_are_not_location_changes(self):
        source = self._relocate()
        original = copy.deepcopy(self.receipt)
        mutations = (
            ("source_file", "source_files", "l0_regions/donor_data.py", "sha256"),
            ("actual_dependency", "actual_dependencies", "l0_regions.donor_learning.validate_rows", "sha256"),
            ("operator_AST", "exact_AST_definition_sha256", "l0_regions/donor_data.py", "assignment"),
        )
        for kind, collection, name, field in mutations:
            self.receipt = copy.deepcopy(original)
            self.receipt[collection][name][field] = "0"*64
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, field):
                self._reopen(source)
            self._assert_cache_preserved()
        self.receipt = copy.deepcopy(original)
        self.receipt["algorithm_rewritten"] = True
        with self.assertRaisesRegex(ValueError, "algorithm_rewritten"):
            self._reopen(source)
        self._assert_cache_preserved()

    def test_real_relocated_source_and_saved_graph_configuration_changes_are_rejected(self):
        source = self._relocate()
        path = Path(self.fixture.modules["sample"].__file__)
        original = path.read_bytes()
        path.write_bytes(original + b"# UNIT different native sampler bytes\n")
        with self.assertRaisesRegex(ValueError, "source_sha256.sample"):
            self._reopen(source)
        path.write_bytes(original)
        config = copy.deepcopy(self.published.config)
        config["graph"]["sample_hops"] = 3
        with self.assertRaisesRegex(ValueError, "config.graph.sample_hops"):
            self._reopen(source, config=config)
        self._assert_cache_preserved()

    def test_inventory_raw_data_binding_and_helper_SHA_are_rejected(self):
        source = self._relocate()
        value = json.loads(self.fixture.input.read_text(encoding="utf8"))
        value["raw_records"][0]["image_sha256"] = "0"*64
        changed_inventory = self.fixture.input.parent / "UNIT_changed_data_index.json"
        changed_inventory.write_text(json.dumps(value), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "input_inventory_sha256"):
            self._reopen(source, inventory=changed_inventory)
        original_sha = geometry_module._sha
        helper = Path(data_contract.__file__).resolve()
        def different_helper(path):
            return "0"*64 if Path(path).resolve() == helper else original_sha(path)
        with patch.object(geometry_module, "_sha", side_effect=different_helper), \
                self.assertRaisesRegex(ValueError, "native_data_contract_sha256"):
            self._reopen(source)
        self._assert_cache_preserved()

    def test_malformed_receipt_locations_missing_and_extra_fields_are_rejected(self):
        source = self._relocate()
        original = copy.deepcopy(self.receipt)
        for kind in ("relative_root", "traversal_root", "wrong_source_path", "wrong_common_path",
                     "root_extra", "source_extra", "dependency_extra", "missing_AST"):
            self.receipt = copy.deepcopy(original)
            if kind == "relative_root":
                self.receipt["source_root"] = "UNIT_relative_root"
            elif kind == "traversal_root":
                self.receipt["source_root"] = str(self.fixture.root / ".." / "UNIT_wrong_root")
            elif kind == "wrong_source_path":
                self.receipt["source_files"]["hiercp_v22/data.py"]["path"] = str(source / "hiercp" / "data.py")
            elif kind == "wrong_common_path":
                self.receipt["actual_dependencies"]["hiercp.common"]["path"] = str(source / "common.py")
            elif kind == "root_extra":
                self.receipt["unexpected"] = True
            elif kind == "source_extra":
                self.receipt["source_files"]["hiercp_v22/data.py"]["unexpected"] = True
            elif kind == "dependency_extra":
                self.receipt["actual_dependencies"]["hiercp.common"]["unexpected"] = True
            else:
                del self.receipt["exact_AST_definition_sha256"]["hiercp_v22/data.py"]["sources"]
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "request differs"):
                self._reopen(source)
            self._assert_cache_preserved()

    def test_relocation_does_not_relax_payload_SHA_or_complete_index_checks(self):
        source = self._relocate()
        row = self.published.rows[-1]
        payload = self.published.output / self.published._canonical[row["id"]]["path"]
        original_payload = payload.read_bytes()
        payload.write_bytes(original_payload + b"UNIT corrupt cache payload")
        with self.assertRaisesRegex(ValueError, "cache file identity/path changed"):
            self._reopen(source)
        payload.write_bytes(original_payload)
        index_path = self.published.output / "index.json"
        original_index = index_path.read_bytes()
        value = json.loads(original_index)
        value["records"] = value["records"][:-1]
        index_path.write_text(json.dumps(value), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "complete exact ordered receipts"):
            self._reopen(source)
        index_path.write_bytes(original_index)
        # Mutations in this one test are exclusively fixture-owned and restored.
        self.assertEqual({name: sha for name, (sha, _) in self.fixture.cache_snapshot(self.published.output).items()},
                         {name: sha for name, (sha, _) in self.before.items()})


if __name__ == "__main__":
    unittest.main()
