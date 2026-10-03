"""Self-contained source/gradient/digest support for the spatial DEBUG probe.

These helpers have no training entrypoint and import no old DEBUG tools. Neural
digest and module-group arithmetic are the same as the recorded v1 smoke paths.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def activate_original(source: Path) -> dict:
    """Verify the pinned archive and every core source before source imports."""
    source = Path(source).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Original-source activation requires a snapshot directory")
    already = [name for name in sys.modules if name == "hiercp" or name.startswith("hiercp.")]
    if already:
        raise RuntimeError("Original-source activation must precede hiercp imports")
    # This has no hiercp import side effect. It checks the pinned archive SHA,
    # pinned manifest, inventory, file SHA/size and original training config.
    from hiercp_v1x.contracts import verify_archive
    proof = verify_archive(ROOT)
    archive = ROOT / "versions/v1/pipeline_v1_source.zip"
    verified = {}
    with ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not ((name.startswith("hiercp/") and name.endswith(".py")) or name == "config/train.json"):
                continue
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError(f"Unsafe archived original module path: {name}")
            target = source.joinpath(*relative.parts)
            if (target.is_symlink() or not target.resolve().is_relative_to(source)
                    or not target.is_file()):
                raise ValueError(f"Original v1 source differs or is absent: {name}")
            expected = hashlib.sha256(bundle.read(name)).hexdigest()
            if _sha(target) != expected:
                raise ValueError(f"Original v1 source differs or is absent: {name}")
            verified[name] = expected
    if len(verified) < 10:
        raise ValueError("Incomplete original v1 archive")
    # An extra importable module must not be allowed to shadow original imports.
    expected_core = {name for name in verified if name.startswith("hiercp/")}
    actual_core = {p.relative_to(source).as_posix() for p in (source / "hiercp").rglob("*.py")}
    if actual_core != expected_core:
        raise ValueError("Unlisted or missing importable original hiercp source files")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(source))
    return dict(source=str(source), archive_sha256=proof["archive_sha256"], verified_files=verified)


def _gradient_groups(model) -> dict:
    """Original nine module groups and complete-loss norm calculation."""
    import torch
    groups = dict(CNN=model.local_encoder.dense_encoder, L0=model.local_encoder.blocks,
                  local_encoder_total=model.local_encoder, role_attention_pool=model.local_encoder.pool,
                  shell_attention_pool=model.local_encoder.context_shell_pool,
                  local_fusion=model.local_encoder.final_fuse, L1=model.patient_encoder,
                  L2=model.prototype_encoder, scalar_score=model.score_head)
    output = {}
    for name, module in groups.items():
        values = [parameter.grad.detach().float().square().sum() for parameter in module.parameters()
                  if parameter.requires_grad and parameter.grad is not None]
        if not values:
            raise AssertionError(f"Complete native loss did not reach {name}")
        output[name] = torch.stack(values).sum().sqrt()
    result = {name: float(value) for name, value in output.items()}
    if any(not 0 < value < float("inf") for value in result.values()):
        raise AssertionError("Zero or nonfinite core-module gradient")
    return result


def state_digest(state):
    """Original neural-byte digest; excludes only sampler identity metadata."""
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name == "v1x_sampling_digest":
            continue
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((str(value.dtype), tuple(value.shape))).encode())
        h.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()
