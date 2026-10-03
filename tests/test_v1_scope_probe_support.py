"""Source identity and tensor-metadata UNIT checks; no neural training run."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch
from torch import nn

from hiercp_v1x import scope_probe_support as support
from hiercp_v1x.contracts import ContractError, V1_ARCHIVE_SHA256


ROOT = Path(__file__).resolve().parents[1]


def unit_directory():
    root = ROOT / "work/scope_probe_support_UNIT"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="metadata_UNIT_", dir=root)


def module_fixture():
    """Parameter containers for gradient-norm arithmetic, no forward method."""
    model = nn.Module()
    local = nn.Module()
    for key in ("dense_encoder", "blocks", "pool", "context_shell_pool", "final_fuse"):
        setattr(local, key, nn.Linear(2, 2))
    model.local_encoder = local
    for key in ("patient_encoder", "prototype_encoder", "score_head"):
        setattr(model, key, nn.Linear(2, 2))
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, index + .5)
    return model


class ScopeDigestUnits(unittest.TestCase):
    def test_digest_exact_original_math_independent_of_mapping_order(self):
        state = {"z": torch.tensor([1., -3.]), "a": torch.arange(12).reshape(3, 4).T}
        h = hashlib.sha256()
        for name in sorted(state):
            value = state[name].detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str((str(value.dtype), tuple(value.shape))).encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        self.assertEqual(support.state_digest(state), h.hexdigest())
        self.assertEqual(support.state_digest(state), support.state_digest(dict(reversed(list(state.items())))))

    def test_only_sampler_identity_is_excluded_and_every_neural_change_is_bound(self):
        original = {"weight": torch.tensor([1., 2.])}
        identity = {**original, "v1x_sampling_digest": torch.arange(32, dtype=torch.uint8)}
        self.assertEqual(support.state_digest(original), support.state_digest(identity))
        for changed in ({"weight": torch.tensor([1., 3.])}, {"other": original["weight"]},
                        {"weight": original["weight"].double()}, {"weight": original["weight"].reshape(1, 2)}):
            with self.subTest(state=list(changed)):
                self.assertNotEqual(support.state_digest(original), support.state_digest(changed))

    def test_state_storage_values_and_noncontiguous_layout_are_not_modified(self):
        value = torch.arange(12.).reshape(3, 4).T
        before = value.clone()
        stride = value.stride()
        support.state_digest({"weight": value})
        self.assertTrue(torch.equal(value, before))
        self.assertEqual(value.stride(), stride)


class ScopeGradientGroupUnits(unittest.TestCase):
    def test_all_original_module_groups_and_norms_exact(self):
        model = module_fixture()
        groups = dict(CNN=model.local_encoder.dense_encoder, L0=model.local_encoder.blocks,
                      local_encoder_total=model.local_encoder, role_attention_pool=model.local_encoder.pool,
                      shell_attention_pool=model.local_encoder.context_shell_pool,
                      local_fusion=model.local_encoder.final_fuse, L1=model.patient_encoder,
                      L2=model.prototype_encoder, scalar_score=model.score_head)
        result = support._gradient_groups(model)
        self.assertEqual(set(result), set(groups))
        self.assertEqual(len(result), 9)
        for name, module in groups.items():
            expected = torch.stack([p.grad.detach().float().square().sum()
                                    for p in module.parameters() if p.requires_grad and p.grad is not None]).sum().sqrt()
            self.assertEqual(result[name], float(expected))

    def test_missing_zero_nan_and_infinite_gradient_are_rejected(self):
        for kind in ("missing", "zero", "nan", "infinite"):
            model = module_fixture()
            for parameter in model.local_encoder.dense_encoder.parameters():
                parameter.grad = None if kind == "missing" else torch.full_like(parameter,
                    {"zero": 0., "nan": float("nan"), "infinite": float("inf")}.get(kind, 0.))
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                support._gradient_groups(model)

    def test_gradient_readout_does_not_modify_any_gradient_or_weight(self):
        model = module_fixture()
        before = [(p.detach().clone(), p.grad.clone()) for p in model.parameters()]
        support._gradient_groups(model)
        for parameter, (value, gradient) in zip(model.parameters(), before):
            self.assertTrue(torch.equal(parameter, value))
            self.assertTrue(torch.equal(parameter.grad, gradient))


class ScopeSourceActivationUnits(unittest.TestCase):
    @staticmethod
    def extract_core(source):
        with ZipFile(ROOT / "versions/v1/pipeline_v1_source.zip") as bundle:
            names = [name for name in bundle.namelist()
                     if (name.startswith("hiercp/") and name.endswith(".py")) or name == "config/train.json"]
            for name in names:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(name))
        return names

    @staticmethod
    def isolated_import_state(stack):
        filtered = {name: value for name, value in sys.modules.items()
                    if not (name == "hiercp" or name.startswith("hiercp."))}
        stack.enter_context(patch.object(support.sys, "modules", filtered))
        stack.enter_context(patch.object(support.sys, "path", list(sys.path)))

    def test_every_original_core_module_and_config_bound_before_import_precedence(self):
        with unit_directory() as name, ExitStack() as stack:
            source = Path(name) / "source"
            names = self.extract_core(source)
            self.isolated_import_state(stack)
            result = support.activate_original(source)
            self.assertEqual(result["archive_sha256"], V1_ARCHIVE_SHA256)
            self.assertEqual(set(result["verified_files"]), set(names))
            self.assertEqual(support.sys.path[:2], [str(source.resolve()), str(ROOT)])
            self.assertFalse(any(key == "hiercp" or key.startswith("hiercp.") for key in support.sys.modules))
            for relative, digest in result["verified_files"].items():
                self.assertEqual(hashlib.sha256((source / relative).read_bytes()).hexdigest(), digest)

    def test_activation_cannot_rebind_an_already_imported_core_package(self):
        with unit_directory() as name, ExitStack() as stack:
            source = Path(name)
            self.isolated_import_state(stack)
            support.sys.modules["hiercp"] = object()
            before = list(support.sys.path)
            with self.assertRaisesRegex(RuntimeError, "precede"):
                support.activate_original(source)
            self.assertEqual(support.sys.path, before)

    def test_changed_missing_or_extra_importable_source_rejected_without_path_mutation(self):
        for kind in ("changed", "missing", "extra"):
            with self.subTest(kind=kind), unit_directory() as name, ExitStack() as stack:
                source = Path(name) / "source"
                self.extract_core(source)
                self.isolated_import_state(stack)
                if kind == "changed":
                    (source / "hiercp/__init__.py").write_text("# UNIT deliberately changed verified copy\n")
                elif kind == "missing":
                    (source / "hiercp/__init__.py").unlink()
                else:
                    (source / "hiercp/unlisted_UNIT.py").write_text("# UNIT never executed\n")
                before = list(support.sys.path)
                with self.assertRaisesRegex(ValueError, "differs|absent|Unlisted"):
                    support.activate_original(source)
                self.assertEqual(support.sys.path, before)

    def test_pinned_archive_validation_failure_cannot_be_ignored(self):
        with unit_directory() as name, ExitStack() as stack:
            source = Path(name) / "source"
            self.extract_core(source)
            self.isolated_import_state(stack)
            verify = stack.enter_context(patch("hiercp_v1x.contracts.verify_archive",
                                               side_effect=ContractError("UNIT archive digest changed")))
            before = list(support.sys.path)
            with self.assertRaisesRegex(ContractError, "digest changed"):
                support.activate_original(source)
            verify.assert_called_once_with(ROOT)
            self.assertEqual(support.sys.path, before)


if __name__ == "__main__":
    unittest.main()
