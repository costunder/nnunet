"""Production STU-Net-S encoder and L0 semantic readout for v2.4.

BasicResBlock and six-stage encoder equations are adapted from STU-Net
by Ziyan Huang et al., Apache-2.0. Upstream:
https://github.com/uni-medical/STU-Net
The encoder checkpoint is the pinned TotalSegmentator-only small_ep4k.model.
No segmentation decoder is instantiated. This module does not import DEBUG tools.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import hashlib
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

CHECKPOINT_URL = "https://huggingface.co/ziyanhuang/STU-Net/resolve/main/small_ep4k.model"

CHECKPOINT_SHA256 = "f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6"

CHANNELS = (16, 32, 64, 128, 256, 256)

DEPTHS = (1, 1, 1, 1, 1, 1)

POOL_STRIDES = ((2, 2, 2),) * 4 + ((1, 1, 2),)

DOWNSAMPLE = tuple(math.prod(stride[axis] for stride in POOL_STRIDES) for axis in range(3))

NUM_SEGMENTATION_CLASSES = 105

class BasicResBlock(nn.Module):
    """Official STU-Net residual block; names preserve checkpoint identity."""

    def __init__(self, input_channels, output_channels, kernel_size=3,
                 padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size,
                               stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size,
                               padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = (nn.Conv3d(input_channels, output_channels, kernel_size=1,
                                stride=stride) if use_1x1conv else None)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class STUNetSmallEncoder(nn.Module):
    """All six official Small stages; full spatial multiscale encoder output."""

    def __init__(self):
        super().__init__()
        self.conv_blocks_context = nn.ModuleList()
        for index, channels in enumerate(CHANNELS):
            incoming = 1 if index == 0 else CHANNELS[index - 1]
            stride = 1 if index == 0 else POOL_STRIDES[index - 1]
            self.conv_blocks_context.append(nn.Sequential(BasicResBlock(
                incoming, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                stride=stride, use_1x1conv=True)))

    def forward(self, x):
        maps = []
        for stage in self.conv_blocks_context:
            if self.training and torch.is_grad_enabled():
                x = activation_checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            maps.append(x)
        return maps

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _decoder_shapes():
    """Exact allowed full-Small decoder tensors, without allocating a decoder."""
    expected = {}
    for stage in range(5):
        incoming, outgoing = CHANNELS[-1-stage], CHANNELS[-2-stage]
        up = f"upsample_layers.{stage}.conv"
        expected[up + ".weight"] = (outgoing, incoming, 1, 1, 1)
        expected[up + ".bias"] = (outgoing,)
        block = f"conv_blocks_localization.{stage}.0"
        for name, input_width, size in (("conv1", 2*outgoing, 3),
                                        ("conv2", outgoing, 3),
                                        ("conv3", 2*outgoing, 1)):
            expected[f"{block}.{name}.weight"] = (outgoing, input_width, size, size, size)
            expected[f"{block}.{name}.bias"] = (outgoing,)
        for name in ("norm1", "norm2"):
            expected[f"{block}.{name}.weight"] = (outgoing,)
            expected[f"{block}.{name}.bias"] = (outgoing,)
        head = f"seg_outputs.{stage}"
        expected[head + ".weight"] = (NUM_SEGMENTATION_CLASSES, outgoing, 1, 1, 1)
        expected[head + ".bias"] = (NUM_SEGMENTATION_CLASSES,)
    return expected

def load_encoder_strict(encoder, checkpoint_path):
    """Require every encoder tensor; exclude only exact known decoder keys."""
    path = Path(checkpoint_path).resolve(strict=True)
    digest = _sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Expected the pinned official STU-Net-S checkpoint SHA256")
    # This exact official envelope has NumPy float32/float64 scalar metadata.
    # No arbitrary Python objects or weights_only=False fallback are permitted.
    allowed_global_names = {"numpy.core.multiarray.scalar", "numpy.dtype"}
    unsafe_globals = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if unsafe_globals - allowed_global_names:
        raise ValueError(f"Unreviewed checkpoint globals: {sorted(unsafe_globals - allowed_global_names)}")
    numpy_scalar = np.core.multiarray.scalar
    allowlist = [(numpy_scalar, "numpy.core.multiarray.scalar"), np.dtype,
                 type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowlist):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Official STU-Net-S checkpoint must contain state_dict")
    expected = encoder.state_dict()
    decoder = _decoder_shapes()
    loaded, ignored, renamed = {}, {}, {}
    for original, value in checkpoint["state_dict"].items():
        if not isinstance(original, str) or not torch.is_tensor(value):
            raise TypeError("Official state_dict must map names to tensors")
        key = original[len("module."):] if original.startswith("module.") else original
        if key in expected:
            if key in loaded:
                raise ValueError(f"Duplicated STU-Net encoder key: {key}")
            target = expected[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"STU-Net encoder shape/dtype mismatch: {original}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Nonfinite STU-Net encoder tensor: {original}")
            loaded[key] = value
            if original != key:
                renamed[original] = key
        elif key in decoder:
            if tuple(value.shape) != decoder[key] or value.dtype != torch.float32:
                raise ValueError(f"Unexpected known STU-Net decoder shape/dtype: {original}")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Nonfinite STU-Net decoder tensor: {original}")
            if key in ignored:
                raise ValueError(f"Duplicated excluded STU-Net decoder key: {key}")
            ignored[key] = dict(shape=list(value.shape),
                                reason="official segmentation decoder/head excluded from encoder-only L0")
        else:
            raise ValueError(f"Unknown STU-Net-S checkpoint tensor: {original}")
    missing = sorted(set(expected) - set(loaded))
    if missing:
        raise ValueError(f"STU-Net-S checkpoint missing encoder tensors: {missing}")
    missing_decoder = sorted(set(decoder) - set(ignored))
    if missing_decoder:
        raise ValueError(f"Not the expected complete official Small checkpoint: {missing_decoder}")
    encoder.load_state_dict(loaded, strict=True)
    parameters = dict(encoder.named_parameters())
    return dict(checkpoint_path=str(path), checkpoint_url=CHECKPOINT_URL,
                sha256=digest, strict=True, weights_only=True,
                numpy_metadata_allowlist=sorted(allowed_global_names) +
                    ["numpy float32 dtype class", "numpy float64 dtype class"],
                expected_encoder_tensors=len(expected), loaded_encoder_tensors=len(loaded),
                loaded_parameter_tensors=len(parameters),
                loaded_parameter_numel=sum(p.numel() for p in parameters.values()),
                tensor_coverage=1.0, parameter_coverage=1.0,
                missing_keys=[], unexpected_keys=[], renamed_keys=renamed,
                ignored_decoder_tensors=ignored, decoder_executed=False,
                scope="all official Small encoder tensors; excludes full segmentation decoder")


@dataclass(frozen=True)
class FeaturePyramid:
    """Full six-scale feature maps with the original dense-view coordinates."""
    maps: tuple[torch.Tensor, ...]
    input_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]

    @property
    def shape(self):
        return (self.maps[0].shape[0], sum(CHANNELS), *self.input_shape)

    def index_select(self, dim, index):
        if dim != 0:
            raise ValueError('Only batch ownership can be indexed in an STU pyramid')
        return FeaturePyramid(tuple(value.index_select(0, index) for value in self.maps),
                              self.input_shape, self.padded_shape)


def sample_feature_scale(feature_map, grid, node_batch, *, input_shape, stride):
    """Vectorized trilinear sampling at each strided convolution's true centers.

    All node counts are retained. Eight interpolation corners are a constant
    tensor dimension; no sample loop, scalar GPU readback, or padded-node cap.
    """
    if feature_map.ndim != 5 or grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError('Explicit [B,C,D,H,W] features and [N,3] xyz grids required')
    if node_batch.ndim != 1 or node_batch.shape[0] != grid.shape[0]:
        raise ValueError('One dense-map owner per graph node required')
    if tuple(input_shape) != tuple(int(v) for v in input_shape) or len(stride) != 3:
        raise ValueError('Original input extent and cumulative stride required')
    n, channels, depth, height, width = feature_map.shape
    owners = node_batch.to(device=feature_map.device, dtype=torch.long)
    torch._assert_async(((owners >= 0) & (owners < n)).all(), 'STU node owner outside batch')
    # stride-2, kernel-3, padding-1 convolution has feature center at i*stride.
    original_xyz = grid.new_tensor(tuple(reversed(input_shape)), dtype=torch.float32)
    stride_xyz = grid.new_tensor(tuple(reversed(stride)), dtype=torch.float32)
    xyz = (grid.float() + 1) * .5 * (original_xyz - 1) / stride_xyz
    maxima = xyz.new_tensor((width - 1, height - 1, depth - 1))
    xyz = torch.minimum(torch.clamp_min(xyz, 0), maxima)
    low = xyz.floor().long()
    high = torch.minimum(low + 1, maxima.long())
    fraction = xyz - low
    corners = torch.tensor(((0,0,0),(1,0,0),(0,1,0),(1,1,0),
                            (0,0,1),(1,0,1),(0,1,1),(1,1,1)),
                           device=xyz.device, dtype=torch.bool)
    locations = torch.where(corners[None], high[:,None], low[:,None])
    weights = torch.where(corners[None], fraction[:,None], 1-fraction[:,None]).prod(-1)
    flat_ids = (((owners[:,None] * depth + locations[...,2]) * height
                 + locations[...,1]) * width + locations[...,0])
    # Channel-last index-select preserves autograd to every sampled feature map.
    flattened = feature_map.permute(0,2,3,4,1).reshape(-1, channels)
    values = flattened.index_select(0, flat_ids.reshape(-1)).reshape(-1, 8, channels)
    return (values * weights.to(values.dtype)[...,None]).sum(1)


class STUNetL0Encoder(nn.Module):
    """Pretrained, trainable STU-S -> physical semantic pools -> twelve L1 fields.

    Replaces the local message-passing encoder. The full official encoder and
    full original physical sampling nodes remain. The original twelve-field
    interface and pooling/fusion widths are preserved for the unchanged L1/L2.
    """
    def __init__(self, original_local_encoder, checkpoint_path):
        super().__init__()
        original = original_local_encoder
        # Forward may carry the admitted empty-context/edge execution wrapper.
        # Architecture constants belong to the unchanged class constructor.
        runtime = type(original).__init__.__globals__
        self.node_types = tuple(runtime['LOCAL_NODE_TYPES'])
        self.source_node_types = frozenset(runtime['SOURCE_LOCAL_NODE_TYPES'])
        self.handcrafted_dim = int(runtime['LOCAL_HANDCRAFTED_DIM'])
        self.hidden_dim = int(original.hidden_dim)
        self.dense_batch_size = int(original.dense_batch_size)
        self.channels_last_3d = bool(original.channels_last_3d)
        if self.hidden_dim != 128 or self.dense_batch_size < 1:
            raise ValueError('Original 128D semantic interface and explicit batch required')
        self.dense_encoder = STUNetSmallEncoder()
        self.load_audit = load_encoder_strict(self.dense_encoder, checkpoint_path)
        if self.channels_last_3d:
            self.dense_encoder.to(memory_format=torch.channels_last_3d)
        self.project = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(self.handcrafted_dim + sum(CHANNELS), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.SiLU(inplace=True)) for name in self.node_types})
        for name in ('pool','context_shell_pool','empty_context_shell','context_shell_fuse',
                     'tumor_fuse','source_context_fuse','target_context_fuse',
                     'source_relation','target_relation','final_fuse'):
            setattr(self, name, getattr(original, name))
        self._pool_context_shells = types.MethodType(type(original)._pool_context_shells, self)
        self._pair = original._pair
        # Same optimizer LR as the GNN arm: the requested backbone is the sole
        # branch difference; all encoder tensors participate in fine-tuning.
        self.config = dict(format='v24_stunet_small_L0_v1', pretrained=True,
            checkpoint_sha256=CHECKPOINT_SHA256, encoder_channels=list(CHANNELS),
            encoder_depths=list(DEPTHS), pool_strides=[list(v) for v in POOL_STRIDES],
            pretrained_dataset='TotalSegmentator 1204 CT / 104 anatomy classes',
            encoder_trainable=True, encoder_lr_scale=1., segmentation_decoder_constructed=False,
            local_message_passing=False, all_six_encoder_scales_used=True,
            original_semantic_readout=True, hidden_dim=self.hidden_dim,
            input_channel='original dense CT channel0 only',
            input_normalization='existing native graph CT clip normalization; encoder fine-tuned',
            input_view='original full-FOV dense view unchanged; end padding only',
            pretrained_preprocessing_identical=False, recipient_tumor_GT_available=False)

    def get_extra_state(self):
        return dict(self.config)

    def set_extra_state(self, state):
        if state != self.config:
            raise ValueError('STU encoder input/weight/architecture contract differs')

    def _encode_dense(self, patches):
        if patches.ndim != 5 or patches.shape[1] != 5 or patches.shape[0] < 1:
            raise ValueError('Full original five-channel dense view required; encoder uses CT only')
        shape = tuple(int(v) for v in patches.shape[2:])
        padded = tuple(math.ceil(v/d)*d for v,d in zip(shape,DOWNSAMPLE))
        if math.prod(v//d for v,d in zip(padded,DOWNSAMPLE)) <= 1:
            padded = (padded[0], padded[1], padded[2] + DOWNSAMPLE[2])
        pad = tuple(value for current,target in reversed(tuple(zip(shape,padded)))
                    for value in (0,target-current))
        ct = patches[:, :1]
        if ct.device.type == 'cpu' or not torch.is_autocast_enabled():
            ct = ct.float()
        ct = F.pad(ct, pad, value=-1.)
        if self.channels_last_3d:
            ct = ct.contiguous(memory_format=torch.channels_last_3d)
        chunks = [self.dense_encoder(ct[start:start+self.dense_batch_size])
                  for start in range(0,len(ct),self.dense_batch_size)]
        maps = tuple(torch.cat([chunk[scale] for chunk in chunks],0) for scale in range(6))
        return FeaturePyramid(maps, shape, padded)

    def encode_dense_maps(self, source_patches, source_graph_index, target_patches):
        if source_graph_index.numel() != target_patches.shape[0]:
            raise ValueError('One source reference for every target observation required')
        source = self._encode_dense(source_patches)
        return source.index_select(0, source_graph_index), self._encode_dense(target_patches)

    def _project_nodes(self, name, raw, grid, owners, pyramid):
        projection = self.project[name]
        linear = projection[0]
        x = F.linear(raw.to(linear.weight.dtype),
                     linear.weight[:, :self.handcrafted_dim], linear.bias)
        offset = self.handcrafted_dim
        stride = (1,1,1)
        for stage, features in enumerate(pyramid.maps):
            if stage:
                stride = tuple(a*b for a,b in zip(stride,POOL_STRIDES[stage-1]))
            sampled = sample_feature_scale(features,grid,owners,
                input_shape=pyramid.input_shape,stride=stride)
            end = offset + CHANNELS[stage]
            x = x + F.linear(sampled,linear.weight[:,offset:end])
            offset = end
        # Split linear projection is mathematically identical to concatenating
        # all752 feature channels but avoids materializing that large tensor.
        return projection[2](projection[1](x))

    def forward_graph(self, batch, source_map, target_map):
        if not isinstance(source_map, FeaturePyramid) or not isinstance(target_map, FeaturePyramid):
            raise TypeError('Complete six-scale STU feature pyramids required')
        x = {}
        for name in self.node_types:
            pyramid = source_map if name in self.source_node_types else target_map
            x[name] = self._project_nodes(name,batch[name].x,batch[name].grid,
                                         batch[name].batch,pyramid)
        pooled = {name:self.pool[name](x[name],index=batch[name].batch,
                    dim_size=source_map.shape[0]) for name in self.pool}
        source_core, source_shells = self._pool_context_shells('source_context',
            x['source_context'],batch['source_context'].x,batch['source_context'].batch,source_map.shape[0])
        target_core, target_shells = self._pool_context_shells('target_context',
            x['target_context'],batch['target_context'].x,batch['target_context'].batch,target_map.shape[0])
        tumor = self.tumor_fuse(self._pair(pooled['tumor_surface'],pooled['tumor_interior']))
        source_context = self.source_context_fuse(self._pair(source_core,pooled['source_liver_surface']))
        target_context = self.target_context_fuse(self._pair(target_core,pooled['target_liver_surface']))
        source_relation = self.source_relation(self._pair(tumor,source_shells[0]))
        target_relation = self.target_relation(self._pair(tumor,target_shells[0]))
        fused = self.final_fuse(torch.cat((tumor,source_context,target_context,
            source_relation,target_relation,torch.abs(source_relation-target_relation)),-1))
        return dict(tumor=tumor,source_context=source_context,target_context=target_context,
            source_relation=source_relation,target_relation=target_relation,
            source_c0=source_shells[0],source_c1=source_shells[1],source_c2=source_shells[2],
            target_c0=target_shells[0],target_c1=target_shells[1],target_c2=target_shells[2],fused=fused)

    def forward(self, batch, source_patches, source_graph_index, target_patches):
        source,target = self.encode_dense_maps(source_patches,source_graph_index,target_patches)
        return self.forward_graph(batch,source,target)


def install_stunet_l0(model, checkpoint_path):
    """Replace registered original L0 entirely; no unused GNN/decoder modules."""
    model.local_encoder = STUNetL0Encoder(model.local_encoder, checkpoint_path)
    model.v24_stunet_contract = dict(model.local_encoder.config)
    return dict(model.local_encoder.load_audit)
