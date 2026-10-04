"""Half A: LocalCNN on preserved v1 fixed dense ROIs, with a role bridge.

This is ``v2CNN_on_preserved_v1_denseROI``, NOT native-spacing v2.2. The
unchanged v1 preparation supplies [B,5,48,48,48] after its adaptive physical
ROI has been resampled, and supplies the original two sampled graph views.
Only CT channel 0 enters the CNN: (-1,1) is converted to (0,1) inside organ
channel 2; outside-organ CT is masked. Footprint channel 1 defines the actual
source tumor and excludes the virtual footprint from context readouts. The
original target erasure, source geometry, candidate pool, curriculum and loss
are outside this module and remain the preserved v1 runner's responsibility.

OrganPyramid (12/24/32 channels, 2/3/3 convolutions), a shared 68->128 readout,
and one shared 512->256->128 donor/recipient difference-product fuser replace
the entire old L0. Full-organ masked scale means supply the LocalCNN fused
pair. The original upper model also requires distinct semantic embeddings:
the bridge samples the pyramid at the ORIGINAL role grids using the ORIGINAL
v1 sampler, then applies mask-weighted role and original physical-shell means.
It does not use local edges, learned graph propagation, handcrafted CT/SDF
columns or the old attention readout. Only x[:,13] identifies the original
shell. A missing sampled shell has an explicit empty-set raw descriptor (zero
before the shared projection), with no dormant learned empty-shell parameter.

The tumor bridge pairs an actual footprint-masked dense descriptor with the
mean of the sampled tumor-surface and tumor-interior descriptors. Context
bridges pair context with its actual liver-surface role. Relation bridges pair
tumor with the corresponding c0 shell. All pairs use [d,r,r-d,r*d]. Thus none
of the upper semantic keys is an alias for the overall fused pair.

Dense execution chunks retain the original size 4. Indexed source maps share
the unique-source storage instead of replicating full 48^3 feature pyramids
per candidate. Chunking processes every ROI/node and changes no physical batch.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from l0_exploration.model import OrganPyramid


MODE = "v2CNN_on_preserved_v1_denseROI"
BRIDGE = "preserved_v1_role_shell_masked_mean_v1"
REVISION = 1
ARCHITECTURE_SUFFIX = f"|half_A_{MODE}_{BRIDGE}_r{REVISION}"
MARKER_KEY = "_half_a_contract_digest"
LOCAL_NODE_TYPES = (
    "tumor_surface", "tumor_interior", "source_context",
    "source_liver_surface", "target_context", "target_liver_surface",
)
SOURCE_ROLES = frozenset(LOCAL_NODE_TYPES[:4])
OPTIONAL_ROLES = frozenset(("source_liver_surface", "target_liver_surface"))
EMBEDDING_KEYS = (
    "tumor", "source_context", "target_context", "source_relation",
    "target_relation", "source_c0", "source_c1", "source_c2",
    "target_c0", "target_c1", "target_c2", "fused",
)


def half_a_spec() -> dict:
    """Serializable mathematics/input identity; execution flags are separate."""
    return {
        "format": "hiercp_v1_half_a_v1", "revision": REVISION,
        "architecture": MODE, "bridge": BRIDGE,
        "input": "preserved_v1_5channel_fixed48_resampled_denseROI",
        "ct": "channel0_minus1_plus1_to_zero_one_only_inside_channel2_organ",
        "footprint": "actual_channel1_mask_not_center_or_sphere",
        "channels": [12, 24, 32], "convolutions": [2, 3, 3],
        "hidden_dim": 128, "readout": "organ_masked_mean_each_scale_shared_project",
        "fusion": "shared_donor_target_signed_difference_product",
        "role_sampling": "original_v1_sampler_original_normalized_role_grids",
        "context_mask": "organ_excluding_actual_or_virtual_footprint",
        "shell_partition": "original_x13_round_times3_clamp0to2",
        "empty_sampled_shell": "zero_raw_mean_before_shared_projection",
        "optional_real_liver_surface_absence": "zero_raw_mean_before_shared_projection_only_if_no_role_nodes",
        "input_admission": "bound_original_10mm_scope_contract_marker_required",
        "tumor": "pair_footprint_mean_with_mean_surface_interior_roles",
        "context": "pair_context_role_with_liver_surface_role",
        "relation": "pair_tumor_with_corresponding_c0_shell",
        "fused": "pair_full_organ_donor_and_full_organ_recipient_means",
        "old_l0_parameters": "replaced_entirely",
        "local_graph_edges": "preserved_input_graph_unused_by_CNN_bridge",
        "native_spacing_v22_equivalence": False,
    }


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def model_contract() -> dict:
    """Human-readable run receipt including all non-native v2.2 differences."""
    return {**half_a_spec(),
            "CNN_input_shape": [5, 48, 48, 48], "CNN_visible_channels": 1,
            "dense_execution_chunk_size": 4, "local_graph_layers_executed": 0,
            "shared_project": [68, 128], "shared_pair_fuser": [512, 256, 128],
            "upper_hidden_dim": 128, "upper_heads": 4,
            "patient_layers": 2, "prototype_layers": 2,
            "original_GT_curriculum_loss": True,
            "training_candidate_count": 8, "candidate_pool_size": 128,
            "target_erasure": "unchanged_original_v1_preparation",
            "source_geometry": "unchanged_actual_original_footprint_and_10mm_scope",
            "source_map_storage": "unique_source_pyramids_with_indexed_graph_owners",
            "differences_from_native_v22_localCNN": [
                "preserved adaptive native ROI resampled to fixed48, not native-spacing crops",
                "original v1 target erasure and five-channel cached payload; only organ CT enters CNN",
                "original sampled roles/shells bridge additional upper semantic inputs",
                "same project and difference-product fuser shared across pair and bridge roles",
                "unchanged original v1 L1/L2/scalar score and original anchor-ranking supervision",
            ]}


def half_a_identity() -> dict:
    """Record current adapter/dependency sources without importing current v1."""
    root = Path(__file__).resolve().parents[1]
    paths = ("hiercp_v1x/half_a_model.py", "l0_exploration/model.py",
             "l0_local_cnn/model.py")
    spec = half_a_spec()
    return {"spec": spec, "contract_sha256": hashlib.sha256(_canonical(spec)).hexdigest(),
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in paths}}


def checkpoint_marker() -> Tensor:
    """CPU serialization marker, copied to the model's device on installation."""
    return torch.tensor(list(hashlib.sha256(_canonical(half_a_spec())).digest()),
                        dtype=torch.uint8)


def require_half_a_state(state: Mapping, *, prefix: str = "") -> None:
    """Reject baseline/other-half state even when a caller requests strict=False."""
    marker = state.get(prefix + MARKER_KEY)
    if (not isinstance(marker, Tensor) or marker.shape != (32,)
            or marker.dtype != torch.uint8
            or not torch.equal(marker.detach().cpu(), checkpoint_marker())):
        raise RuntimeError("Checkpoint does not identify the preserved-v1 half-A CNN/role bridge")


def _reject_other_half(module, state_dict, prefix, local_metadata, strict,
                       missing_keys, unexpected_keys, error_msgs):
    require_half_a_state(state_dict, prefix=prefix)


def _require(value: Tensor, message: str) -> None:
    # Runtime validation must not transfer CUDA tensors to the CPU.
    if value.device.type == "cuda":
        torch._assert_async(value, message)
    elif not bool(value):
        raise ValueError(message)


def organ_ct(patches: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Read the preserved v1 masks and convert CT without changing geometry."""
    if (patches.ndim != 5 or patches.shape[1:] != (5, 48, 48, 48)
            or patches.shape[0] < 1 or not patches.is_floating_point()):
        raise ValueError("Half A requires nonempty preserved v1 floating [B,5,48,48,48] ROIs")
    organ = patches[:, 2:3] > 0.5
    footprint = patches[:, 1:2] > 0.5
    _require(torch.isfinite(patches[:, :3]).all(), "Nonfinite preserved v1 CT/masks")
    _require(organ.flatten(1).any(1).all(), "Empty dense-ROI organ mask")
    _require(footprint.flatten(1).any(1).all(), "Actual/virtual dense-ROI footprint vanished")
    _require((organ & footprint).flatten(1).any(1).all(),
             "Dense-ROI footprint has no organ support")
    ct = patches[:, :1]
    _require(((ct >= -1.001) & (ct <= 1.001) | ~organ).all(),
             "Preserved v1 internal CT is not normalized to [-1,1]")
    # No outside-organ value is converted into visible 0.5 background.
    return torch.where(organ, (ct + 1) * 0.5, 0), organ, footprint


@dataclass(frozen=True)
class PyramidMaps:
    """Unique ROI maps plus graph->ROI ownership for the original v1 API.

    .index_select(0, owners) changes ownership only, retaining one donor CNN
    graph/storage. .shape[0] is the logical graph count, as frozen v1 expects.
    """
    scales: tuple[tuple[Tensor, Tensor], ...]
    organ: Tensor
    footprint: Tensor
    organ_mean: Tensor
    footprint_mean: Tensor
    owners: Tensor

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.owners.numel(), self.organ_mean.shape[1]))

    @property
    def device(self) -> torch.device:
        return self.organ.device

    def index_select(self, dim: int, index: Tensor) -> "PyramidMaps":
        if dim != 0 or index.ndim != 1 or index.dtype != torch.long or index.device != self.device:
            raise ValueError("Pyramid ownership selection requires device-local int64 [N] on dimension 0")
        return replace(self, owners=self.owners.index_select(0, index))


class HalfALocalCNN(nn.Module):
    """All trainable parameters participate in the shared CNN/readout/fusion."""

    def __init__(self, *, sampler: Callable, dropout: float,
                 dense_batch_size: int = 4, channels_last_3d: bool = True,
                 checkpoint_dense_encoder: bool = True, scope_contract: str | None = None):
        super().__init__()
        if dense_batch_size != 4:
            raise ValueError("Half A preserves original dense execution chunk size 4")
        if not callable(sampler) or not 0 <= dropout < 1:
            raise ValueError("Original v1 sampler and valid original dropout required")
        self.hidden_dim = 128
        self.dense_batch_size = dense_batch_size
        self.channels_last_3d = bool(channels_last_3d)
        self.checkpoint_dense_encoder = bool(checkpoint_dense_encoder)
        self.scope_contract = scope_contract
        if scope_contract is not None and re.fullmatch(r"[0-9a-f]{64}", scope_contract) is None:
            raise ValueError("Invalid original bounded-scope contract SHA256")
        self._sampler = sampler
        self.cnn = OrganPyramid([12, 24, 32], self.checkpoint_dense_encoder)
        self.project = nn.Sequential(nn.Linear(68, 128), nn.LayerNorm(128), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(256, 128))
        if self.channels_last_3d:
            self.cnn.to(memory_format=torch.channels_last_3d)

    def get_extra_state(self):
        return {**half_a_spec(), "bounded_scope_contract": self.scope_contract}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("Half-A local checkpoint has a different input/role bridge contract")

    @staticmethod
    def _pair(donor: Tensor, recipient: Tensor) -> Tensor:
        return torch.cat((donor, recipient, recipient - donor, recipient * donor), dim=-1)

    @staticmethod
    def _dense_mean(features: Tensor, weights: Tensor, *, required: bool) -> Tensor:
        count = weights.float().sum((2, 3, 4))
        if required:
            _require((count > 0).all(), "Organ vanished at a CNN scale")
        # An unsupported coarse tumor scale is an absent support, not a failed
        # forward. The fine scale must contain the actual footprint. Area
        # weights preserve sub-stride footprint mass instead of nearest erasure.
        total = (features.float() * weights.float()).sum((2, 3, 4))
        return (total / count.clamp_min(1e-8)).to(features.dtype)

    def _encode_dense(self, patches: Tensor) -> PyramidMaps:
        images, organ, footprint = organ_ct(patches)
        reference = next(self.parameters())
        if patches.device != reference.device:
            raise ValueError("Dense ROIs must already reside on the half-A parameter device")
        if not torch.is_autocast_enabled(patches.device.type):
            images = images.to(reference.dtype)
        if self.channels_last_3d:
            images = images.contiguous(memory_format=torch.channels_last_3d)
        chunks = []
        organ_means, footprint_means = [], []
        for start in range(0, len(images), self.dense_batch_size):
            stop = start + self.dense_batch_size
            maps = self.cnn(images[start:stop], organ[start:stop])
            chunks.append(maps)
            whole, tumor = [], []
            for features, mask in maps:
                whole.append(self._dense_mean(features, mask, required=True))
                mass = F.adaptive_avg_pool3d(footprint[start:stop].float(), features.shape[2:])
                tumor.append(self._dense_mean(features, mass * mask, required=False))
            organ_means.append(torch.cat(whole, dim=1))
            footprint_means.append(torch.cat(tumor, dim=1))
        scales = tuple((torch.cat([chunk[level][0] for chunk in chunks], dim=0),
                        torch.cat([chunk[level][1] for chunk in chunks], dim=0))
                       for level in range(3))
        return PyramidMaps(scales, organ, footprint,
                           torch.cat(organ_means, dim=0), torch.cat(footprint_means, dim=0),
                           torch.arange(len(images), dtype=torch.long, device=images.device))

    def encode_dense_maps(self, source_patches: Tensor, source_graph_index: Tensor,
                          target_patches: Tensor) -> tuple[PyramidMaps, PyramidMaps]:
        if (source_graph_index.ndim != 1 or source_graph_index.dtype != torch.long
                or source_graph_index.numel() != target_patches.shape[0]):
            raise ValueError("One original source owner per target graph is required")
        source = self._encode_dense(source_patches)
        return source.index_select(0, source_graph_index), self._encode_dense(target_patches)

    def _sample(self, maps: PyramidMaps, values: Tensor, grid: Tensor,
                node_batch: Tensor, *, optional: bool = False) -> Tensor:
        """Call the original sampler on shared ROI storage and restore node order."""
        if node_batch.ndim != 1 or node_batch.dtype != torch.long:
            raise ValueError("Every original semantic role requires graph-major nodes")
        if grid.ndim != 2 or grid.shape != (node_batch.numel(), 3) or not grid.is_floating_point():
            raise ValueError("Original role grids must be floating [N,3] and match node ownership")
        if not node_batch.numel():
            if optional:
                # Exactly the admitted bounded-scope empty role, not an error
                # fallback. There are no nodes to interpolate or discard.
                return values.new_empty((0, values.shape[1]))
            raise ValueError("Original mandatory semantic role contains no nodes")
        graph_count = maps.shape[0]
        _require((node_batch >= 0).all() & (node_batch < graph_count).all(),
                 "Local node ownership is outside dense map graph count")
        counts = torch.bincount(node_batch, minlength=graph_count)
        if not optional:
            _require((counts > 0).all(), "Original semantic role missing in a local graph")
        # Some inference chunks use only a subset of unique sources. The native
        # sampler requires every map to own nodes, so select precisely those
        # maps, remap owners, and sort nodes without discarding any node.
        selected_owners = maps.owners[node_batch]
        unique, roi_batch = torch.unique(selected_owners, sorted=True, return_inverse=True)
        order = torch.argsort(roi_batch, stable=True)
        sampled = self._sampler(values.index_select(0, unique), grid[order], roi_batch[order])
        return torch.empty_like(sampled).index_copy(0, order, sampled)

    @staticmethod
    def _role_mean(values: Tensor, weights: Tensor, owner: Tensor,
                   graph_count: int, *, role: str, shell: Tensor | None = None,
                   optional: bool = False) -> Tensor:
        groups = graph_count if shell is None else graph_count * 3
        index = owner if shell is None else owner * 3 + shell
        mass = torch.zeros((groups, 1), device=values.device, dtype=torch.float32)
        total = torch.zeros((groups, values.shape[1]), device=values.device, dtype=torch.float32)
        mass.index_add_(0, index, weights.float())
        total.index_add_(0, index, values.float() * weights.float())
        if shell is None:
            if optional:
                nodes = torch.bincount(index, minlength=groups)
                _require(((mass[:, 0] > 0) | (nodes == 0)).all(),
                         f"Present optional {role} nodes have no dense organ support")
            else:
                _require((mass > 0).all(), f"No dense organ/footprint support for {role}")
        else:
            nodes = torch.bincount(index, minlength=groups)
            _require(((mass[:, 0] > 0) | (nodes == 0)).all(),
                     f"Sampled {role} shell nodes have no dense organ support")
        # Empty sampled shells are explicit structural absence in v1 views.
        output = (total / mass.clamp_min(1e-8)).to(values.dtype)
        return output if shell is None else output.reshape(graph_count, 3, -1)

    def _role(self, maps: PyramidMaps, store, role: str) -> tuple[Tensor, Tensor | None]:
        grid, owner = store.grid, store.batch
        if grid.device != maps.device or owner.device != maps.device:
            raise ValueError("Original role graph and dense ROIs must share a device")
        optional = role in OPTIONAL_ROLES
        sampled = torch.cat([self._sample(maps, features, grid, owner, optional=optional)
                             for features, _ in maps.scales], dim=-1)
        organ = self._sample(maps, maps.organ.to(sampled.dtype), grid, owner, optional=optional)
        footprint = self._sample(maps, maps.footprint.to(sampled.dtype), grid, owner, optional=optional)
        weights = organ * (footprint if role.startswith("tumor_") else (1 - footprint))
        descriptor = self._role_mean(sampled, weights, owner, maps.shape[0], role=role, optional=optional)
        shells = None
        if role.endswith("context"):
            if store.x.ndim != 2 or store.x.shape[1] != 16:
                raise ValueError("Original sixteen-column local role table required for shell partition")
            shell = (store.x[:, 13].float() * 3).round().long().clamp(0, 2)
            shells = self.project(self._role_mean(sampled, weights, owner, maps.shape[0],
                                                 role=role, shell=shell))
        return self.project(descriptor), shells

    def _require_scope(self, batch, graph_count: int) -> None:
        if self.scope_contract is None:
            raise ValueError("Half-A execution requires the preserved original 10mm bounded scope")
        marker = batch.get("v1x_bounded_scope_contract")
        markers = marker if isinstance(marker, (list, tuple)) else [marker]
        if (not markers or any(value != self.scope_contract for value in markers)
                or len(markers) != graph_count):
            raise ValueError("Half-A local graph is native, mixed, or belongs to another bounded scope")
        margin = batch.get("v1x_bounded_scope_margin_mm")
        if isinstance(margin, Tensor):
            _require((margin == 10).all(), "Half-A graph has a different physical scope margin")
            if margin.numel() != graph_count:
                raise ValueError("One original10mm margin marker per local graph required")
        else:
            margins = margin if isinstance(margin, (list, tuple)) else [margin]
            if len(margins) != graph_count or any(value != 10 for value in margins):
                raise ValueError("One original10mm margin marker per local graph required")

    def forward_graph(self, batch, source_map: PyramidMaps,
                      target_map: PyramidMaps) -> dict[str, Tensor]:
        if not isinstance(source_map, PyramidMaps) or not isinstance(target_map, PyramidMaps):
            raise TypeError("Half-A forward_graph requires its encoded PyramidMaps")
        if source_map.shape[0] != target_map.shape[0] or set(batch.node_types) != set(LOCAL_NODE_TYPES):
            raise ValueError("Preserved six-role graph and matching source/target owners required")
        self._require_scope(batch, source_map.shape[0])
        roles, shells = {}, {}
        for role in LOCAL_NODE_TYPES:
            maps = source_map if role in SOURCE_ROLES else target_map
            roles[role], shell_values = self._role(maps, batch[role], role)
            if shell_values is not None:
                shells[role] = shell_values
        donor = self.project(source_map.organ_mean[source_map.owners])
        recipient = self.project(target_map.organ_mean[target_map.owners])
        footprint = self.project(source_map.footprint_mean[source_map.owners])
        tumor_roles = (roles["tumor_surface"] + roles["tumor_interior"]) * 0.5
        tumor = self.fuse(self._pair(footprint, tumor_roles))
        source_context = self.fuse(self._pair(roles["source_context"], roles["source_liver_surface"]))
        target_context = self.fuse(self._pair(roles["target_context"], roles["target_liver_surface"]))
        output = {
            "tumor": tumor, "source_context": source_context, "target_context": target_context,
            "source_relation": self.fuse(self._pair(tumor, shells["source_context"][:, 0])),
            "target_relation": self.fuse(self._pair(tumor, shells["target_context"][:, 0])),
            "fused": self.fuse(self._pair(donor, recipient)),
        }
        for branch in ("source", "target"):
            for index in range(3):
                output[f"{branch}_c{index}"] = shells[f"{branch}_context"][:, index]
        for key, value in output.items():
            if value.shape != (source_map.shape[0], 128):
                raise RuntimeError(f"Half-A {key} lost the original graph/128D contract")
            _require(torch.isfinite(value).all(), f"Nonfinite half-A {key} embedding")
        return {key: output[key] for key in EMBEDDING_KEYS}

    def forward(self, batch, source_patches: Tensor, source_graph_index: Tensor,
                target_patches: Tensor) -> dict[str, Tensor]:
        source, target = self.encode_dense_maps(source_patches, source_graph_index, target_patches)
        return self.forward_graph(batch, source, target)


def install_half_a(model: nn.Module) -> nn.Module:
    """Replace only a constructed preserved-v1 L0, before optimizer creation.

    The caller owns original-source activation. The replacement is freshly
    initialized with seed42 in a forked PyTorch RNG context; installing it does
    not advance the original CPU/allocated-CUDA dropout/loader RNG streams.
    Existing L1/L2/readout/scalar parameters and all upper modules retain their
    objects/values. No baseline L0 layer is kept frozen or disconnected.
    """
    if isinstance(model.local_encoder, HalfALocalCNN):
        return model
    old = model.local_encoder
    original_module = sys.modules.get(type(old).__module__)
    sampler = getattr(original_module, "sample_dense_features_variable", None)
    if (getattr(model, "hidden_dim", None) != 128 or getattr(old, "hidden_dim", None) != 128
            or getattr(model, "ablation_mode", None) != "full"
            or len(getattr(old, "project", {})) != 6 or len(getattr(old, "blocks", ())) != 3
            or old.dense_batch_size != 4 or sampler is None):
        raise ValueError("Half A requires the complete original v1 128D/three-layer/six-role model")
    for name in ("patient_encoder", "prototype_encoder"):
        blocks = getattr(model, name).blocks
        if len(blocks) != 2 or any(conv.heads != 4 for block in blocks for conv in block.conv.convs.values()):
            raise ValueError("Half A preserves original 128D/four-head/two+two upper layers")
    if getattr(model, "v1x_stage", None) is not None:
        raise ValueError("Install half A into a fresh preserved v1 model, not another v1.x L0 stage")
    scope = re.search(r"\|bounded_scope_([0-9a-f]{64})(?:\||$)", model.architecture_version)
    if scope is None:
        raise ValueError("Half A requires the preserved original bounded10mm scope identity")
    parameter = next(old.parameters())
    dropout = old.final_fuse[3].p
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    with torch.random.fork_rng(devices=devices):
        # Initialization runs on CPU. Seed only that generator: manual_seed()
        # would also reseed unrelated visible CUDA generators for a CPU model.
        torch.random.default_generator.manual_seed(42)
        replacement = HalfALocalCNN(sampler=sampler, dropout=dropout,
                                   dense_batch_size=old.dense_batch_size,
                                   channels_last_3d=old.channels_last_3d,
                                   checkpoint_dense_encoder=old.checkpoint_dense_encoder,
                                   scope_contract=scope.group(1))
        replacement.to(device=parameter.device, dtype=parameter.dtype)
    replacement.train(old.training)
    model.local_encoder = replacement
    model.half_a_mode = MODE
    model.half_a_bridge = BRIDGE
    model.architecture_version = model.architecture_version + ARCHITECTURE_SUFFIX
    model.register_buffer(MARKER_KEY, checkpoint_marker().to(parameter.device))
    model.register_load_state_dict_pre_hook(_reject_other_half)
    return model
