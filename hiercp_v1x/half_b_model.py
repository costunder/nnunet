"""Half B: original v1 L0 -> tracked legacy v2.2 prompt/cluster/cosine upper.

The ORIGINAL placement model's forward, local encoder, two graph views and
six-term view consistency remain in use. Its genuine ``local['fused']``128D
pair vector is the bridge: original tumor/context/shell/relation readouts all
already contribute to that vector. No replicated semantic pseudo-nodes, query
GT labels, patient raw shortcuts, or new trainable bridge are introduced.

The five original upper/scalar modules are removed. The actual tracked
``hiercp_v222.model.PromptGraphModel`` supplies legacy RelationLayer x2,
cross-patient MultiheadAttention/Residual x2 and class-conditional clustered
cosine prototypes. The production LocalCNN runner/recommender's scalar is
``logit1-logit0``; that same scalar feeds the unchanged v1 ranking objective.
Native observation CE/alignment CE are NOT appended to the original v1 loss.
Consequently this is a declared architecture bridge, not native-v2.2 equivalence.

Every support row is a real original training-cache candidate: index0 is class1
and indices1..7 are class0. Full training-only, fixed-epoch0, two-view detached
L0 memory is explicitly supplied by the runner. It is refreshed initially and
after each training epoch before validation. Teacher plans are lazy once per
refresh/excluded query patient; upper states are recomputed LIVE each forward.

Mixed query patients keep the original physical batch. Their complete eligible
support episodes run in one disjoint L1 call and padded batched L2; only invalid
padding and same-patient attention are masked. Each query reads its own episode
labels/prototypes, without reverse query edges. All eligible support is retained.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


MODE = "preserved_v1_L0_with_tracked_v22_legacy_prompt_cluster_cosine_upper"
REVISION = 1
MARKER_KEY = "_half_b_contract_digest"
ARCHITECTURE_SUFFIX = f"|half_B_{MODE}_r{REVISION}"
SUPPORT_POLICY = "fixed_epoch0_two_original_views_eval_no_grad_full_train_initial_then_after_each_train_before_validation"
_REMOVED_UPPER = ("patient_encoder", "prototype_encoder", "patient_readout",
                  "population_readout", "score_head")
_TRACKED_SOURCES = {
    "hiercp_v222/model.py": "53d4d95220eab2ba643424a3e161fac4b91ba6a44f8ec7a108fa44dd9dce0a7d",
    "hiercp_v222/clustering.py": "a7d73761eeecadcda9ef0b2e0657e62068d260af42b1d519103fa0d1ddf211fc",
}


def half_b_spec() -> dict:
    return {
        "format": "hiercp_v1_half_b_v1", "revision": REVISION, "architecture": MODE,
        "local_encoder": "actual_original_v1_object_state_and_forward_preserved",
        "bridge": "genuine_original_v1_semantic_pair_fused128_no_new_projection",
        "legacy_upper": "hiercp_v222.model.PromptGraphModel",
        "hidden_dim": 128, "heads": 4, "task_layers": 2, "alignment_layers": 2,
        "temperature": 0.2, "native_alignment_loss_weight_metadata": 1,
        "L1": "legacy_RelationLayer_edge_is_support_true_relation_not_official_reference_L1",
        "L2": "same_patient_masked_cross_patient_MHA_Residual_two_layers",
        "clusters": "class_conditional_cosine_average_linkage_max_positive_silhouette_all_non_singleton_cuts_else_one",
        "scorer": "native_patient_mass_weighted_cosine_mixture_logit1_minus_logit0",
        "support_policy": SUPPORT_POLICY, "fixed_support_view_epoch": 0,
        "support_classes": "original_v1_training_anchor_index0_class1_curriculum_indices1to7_class0",
        "support_exclusion": "entire_query_patient_from_both_original_source_and_recipient_support",
        "query_GT_in_forward": False, "query_to_support_reverse_edges": False,
        "support_embeddings_gradient": "detached_eval_no_grad_L0",
        "query_L0_gradient": "live_original_forward",
        "upper_gradient": "live_L1_L2_cosine_every_forward",
        "teacher_plan": "detached_eval_L1_lazy_once_per_refresh_and_exclusion_key",
        "batching": "complete_disjoint_L1_episodes_batched_masked_L2_and_episode_masked_cosine_mixtures",
        "loss": "unchanged_original_v1_curriculum_ranking_plus_original_two_view_consistency",
        "native_observation_CE_added": False, "native_alignment_CE_added": False,
        "native_v22_equivalence": False,
        "removed_original_upper": list(_REMOVED_UPPER),
        "tracked_upper_source_sha256": dict(_TRACKED_SOURCES),
    }


def model_contract() -> dict:
    return {**half_b_spec(), "original_input_shape": [5, 48, 48, 48],
            "original_local_layers": 3, "original_dense_execution_chunk_size": 4,
            "training_candidates_per_sample": 8, "candidate_pool_size": 128,
            "configured_train_cases": 84, "configured_validation_cases": 21,
            "production_epochs": 40, "upper_only_initialization": "fresh_seed42_forked_RNG",
            "differences_from_native_v22": [
                "original v1 L0, five-channel fixed48 ROI and target erasure are preserved",
                "original source-anchor/curriculum support classes replace observed-tumor P/U classes",
                "original ranking and view consistency replace native observation/alignment CE objectives",
                "support is complete original training-cache anchor/curriculum candidates",
                "prompt teacher plans are bound to the explicit original support refresh policy",
            ]}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def half_b_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = ("hiercp_v1x/half_b_model.py", *_TRACKED_SOURCES,
             "l0_regions/donor_learning.py", "l0_local_cnn/recommendation.py")
    return {"spec": half_b_spec(), "contract_sha256": hashlib.sha256(_canonical(half_b_spec())).hexdigest(),
            "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                              for path in paths}}


def checkpoint_marker(*, debug_support=False) -> Tensor:
    digest = hashlib.sha256(_canonical({**half_b_spec(), "debug_support": bool(debug_support)})).digest()
    return torch.tensor(list(digest), dtype=torch.uint8, device="cpu")


def require_half_b_state(state, *, prefix="", debug_support=False) -> None:
    marker = state.get(prefix + MARKER_KEY) if isinstance(state, Mapping) else None
    if (not isinstance(marker, Tensor) or marker.dtype != torch.uint8 or marker.shape != (32,)
            or not torch.equal(marker.detach().cpu(), checkpoint_marker(debug_support=debug_support))):
        raise RuntimeError("Checkpoint is not this half-B legacy upper/support profile")


def _reject_other_half(module, state, prefix, local_metadata, strict,
                       missing_keys, unexpected_keys, error_msgs):
    require_half_b_state(state, prefix=prefix, debug_support=module.half_b.debug_support)


def _require(condition: Tensor, message: str) -> None:
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _strings(values, label: str, *, unique=False) -> tuple[str, ...]:
    if (not isinstance(values, (tuple, list)) or not values
            or any(not isinstance(value, str) or not value for value in values)
            or unique and len(set(values)) != len(values)):
        raise ValueError(f"Nonempty {'unique ' if unique else ''}{label} string metadata required")
    return tuple(values)


def _tensor_signature(tensor: Tensor) -> tuple:
    return (id(tensor), tensor._version, tuple(tensor.shape), tensor.dtype,
            tensor.device, tensor.requires_grad)


def _to_device(value, device):
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


class HalfBUpper(nn.Module):
    def __init__(self, *, dropout: float, scope_contract: str, debug_support=False):
        super().__init__()
        if re.fullmatch(r"[0-9a-f]{64}", scope_contract) is None:
            raise ValueError("Actual original10mm bounded-scope SHA256 required")
        root = Path(__file__).resolve().parents[1]
        for path, expected in _TRACKED_SOURCES.items():
            if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Tracked legacy upper source changed: {path}")
        # Lazy import is essential: receipt/spec inspection must not load the
        # current canonical hiercp before the archived source is activated.
        native = importlib.import_module("hiercp_v222.model")
        cfg = {"temperature": .2, "alignment_loss_weight": 1,
               "task_layers": 2, "alignment_layers": 2}
        base = {"model": {"hidden_dim": 128, "heads": 4, "dropout": dropout}}
        self.core = native.PromptGraphModel(cfg, base, {}, local_encoder=nn.Identity())
        del self.core.local  # Upper-only owner: no unused/frozen L0 parameters.
        self.scope_contract = scope_contract
        self.debug_support = bool(debug_support)
        self._memory = None
        self._memory_signatures = None
        self._plans = {}
        self._episodes = {}
        self._seen_generations = set()

    def get_extra_state(self):
        return {**half_b_spec(), "bounded_scope_contract": self.scope_contract,
                "debug_support": self.debug_support}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("Half-B checkpoint has another scope/upper/support recipe")
        # A model checkpoint is never mislabeled as a fresh support memory.
        self.clear_support()

    def clear_support(self) -> None:
        self._memory = self._memory_signatures = None
        self._plans = {}
        self._episodes = {}

    def bind_support(self, memory: Mapping) -> dict:
        """Admit one complete, detached and source-bound training-only refresh."""
        if not isinstance(memory, Mapping):
            raise ValueError("Explicit full training support metadata mapping required")
        required = {"embeddings", "owners", "classes", "patient_case_ids", "sample_ids",
                    "candidate_indices", "expected_samples", "training_case_ids", "validation_case_ids",
                    "manifest_sha256", "generation", "epoch", "fixed_view_epoch", "support_policy",
                    "debug", "full_signed_training_cache"}
        if required - set(memory):
            raise ValueError(f"Support metadata incomplete: {sorted(required - set(memory))}")
        embeddings, owners, classes = (memory[key] for key in ("embeddings", "owners", "classes"))
        device = next(self.parameters()).device
        if (not isinstance(embeddings, Tensor) or embeddings.ndim != 2 or embeddings.shape[1] != 128
                or embeddings.dtype != torch.float32 or embeddings.requires_grad or not len(embeddings)
                or embeddings.device != device):
            raise ValueError("Detached complete FP32 [N,128] support on the model device required")
        for value in (owners, classes):
            if (not isinstance(value, Tensor) or value.dtype != torch.long
                    or value.shape != (len(embeddings),) or value.device != device):
                raise ValueError("Support owner/class int64 [N] on the embedding device required")
        patients = _strings(memory["patient_case_ids"], "support patient", unique=True)
        train = _strings(memory["training_case_ids"], "original training split", unique=True)
        validation = _strings(memory["validation_case_ids"], "original validation split", unique=True)
        samples = _strings(memory["sample_ids"], "signed source sample")
        indices = memory["candidate_indices"]
        expected = memory["expected_samples"]
        if (not isinstance(expected, Mapping) or not expected
                or any(not isinstance(key, str) or not key or not isinstance(case, str) or not case
                       for key, case in expected.items())
                or not isinstance(indices, (list, tuple)) or len(indices) != len(embeddings)
                or len(samples) != len(embeddings)
                or any(type(index) is not int or not 0 <= index < 8 for index in indices)):
            raise ValueError("Complete original signed source samples with all eight candidate indices required")
        if (set(train) & set(validation) or not set(patients) <= set(train)
                or set(patients) != set(expected.values()) or len(patients) < 3):
            raise ValueError("At least three actual training-only support patients and disjoint validation required")
        if type(memory["debug"]) is not bool or type(memory["full_signed_training_cache"]) is not bool:
            raise ValueError("Explicit full-cache/DEBUG support admission flags required")
        if self.debug_support:
            if not memory["debug"] or memory["full_signed_training_cache"]:
                raise ValueError("DEBUG upper requires an explicitly labeled DEBUG support fixture")
        elif (memory["debug"] or not memory["full_signed_training_cache"]
              or len(train) != 84 or len(validation) != 21):
            raise ValueError("Production support must cover the signed actual train cache within original84/21 split")
        if (memory["support_policy"] != SUPPORT_POLICY or memory["fixed_view_epoch"] != 0
                or type(memory["epoch"]) is not int or not 0 <= memory["epoch"] <= 40
                or not isinstance(memory["manifest_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", memory["manifest_sha256"]) is None
                or not isinstance(memory["generation"], str) or not memory["generation"]
                or memory["generation"] in self._seen_generations):
            raise ValueError("Fresh unique-generation fixed-epoch0 support recipe/manifest binding required")
        # One intentional small metadata read per full refresh. It never reads
        # query targets or moves live query/GNN activations to CPU.
        metadata = torch.stack((owners, classes), dim=1).detach().cpu().tolist()
        records = Counter()
        for sample, candidate, (owner, cls) in zip(samples, indices, metadata):
            if (not 0 <= owner < len(patients) or sample not in expected
                    or expected[sample] != patients[owner] or cls != int(candidate == 0)):
                raise ValueError("Support source/owner/anchor class provenance differs from original training cache")
            records[(sample, candidate)] += 1
        if (len(embeddings) != len(expected) * 8 or
                records != Counter({(sample, index): 1 for sample in expected for index in range(8)})):
            raise ValueError("Support lost, duplicated or substituted an original training sample/candidate")
        _require(torch.isfinite(embeddings).all(), "Nonfinite full support embeddings")
        # Copy immutable metadata; tensor objects/storage are checked for later
        # mutation before every forward rather than silently trusting a cache.
        self._memory = {**copy.deepcopy({key: value for key, value in memory.items()
                                        if key not in ("embeddings", "owners", "classes")}),
                        "embeddings": embeddings, "owners": owners, "classes": classes,
                        "patient_case_ids": patients, "sample_ids": samples,
                        "candidate_indices": tuple(indices), "expected_samples": dict(expected),
                        "training_case_ids": train, "validation_case_ids": validation}
        self._memory_signatures = tuple(_tensor_signature(value) for value in (embeddings, owners, classes))
        self._seen_generations.add(memory["generation"])
        self._plans = {}
        self._episodes = {}
        return self.support_receipt()

    def _require_memory(self):
        if self._memory is None:
            raise RuntimeError("Half B requires an explicitly refreshed complete training-only support bank")
        values = tuple(self._memory[key] for key in ("embeddings", "owners", "classes"))
        if tuple(_tensor_signature(value) for value in values) != self._memory_signatures:
            raise RuntimeError("Bound support tensor changed after admission; rebuild a full fresh generation")
        if values[0].device != next(self.parameters()).device:
            raise RuntimeError("Bound support must be rebuilt after model device movement")
        return self._memory

    def support_receipt(self) -> dict:
        memory = self._require_memory()
        return {"generation": memory["generation"], "epoch": memory["epoch"],
                "manifest_sha256": memory["manifest_sha256"], "support_policy": SUPPORT_POLICY,
                "fixed_view_epoch": 0, "support_candidates": len(memory["embeddings"]),
                "support_source_samples": len(memory["expected_samples"]),
                "support_patients": list(memory["patient_case_ids"]),
                "debug": memory["debug"], "full_signed_training_cache": memory["full_signed_training_cache"],
                "cached_exclusion_teacher_plans": len(self._plans),
                "query_GT_labels_used": False, "detached_L0_memory": True, "live_upper_states_cached": False}

    def _episode(self, key):
        if key in self._episodes:
            return self._episodes[key]
        memory = self._require_memory()
        patients = memory["patient_case_ids"]
        owner = memory["owners"]
        if key is None:
            keep = torch.ones_like(owner, dtype=torch.bool)
            used = list(range(len(patients)))
        else:
            excluded = patients.index(key)
            keep = owner != excluded
            used = [index for index in range(len(patients)) if index != excluded]
        if len(used) < 2:
            raise ValueError("Two other support patient tasks required after complete query-patient exclusion")
        mapping = torch.full((len(patients),), -1, device=owner.device, dtype=torch.long)
        used_ids = torch.tensor(used, device=owner.device)
        mapping[used_ids] = torch.arange(len(used), device=owner.device)
        ids = keep.nonzero().flatten()
        episode = {"key": key, "indices": ids, "owners": mapping[owner[ids]],
                   "classes": memory["classes"][ids], "patients": len(used),
                   "patient_case_ids": tuple(patients[index] for index in used)}
        self._episodes[key] = episode
        return episode

    def _encode_episodes(self, episodes):
        memory = self._require_memory()
        ids = torch.cat([episode["indices"] for episode in episodes])
        offset, owners = 0, []
        for episode in episodes:
            owners.append(episode["owners"] + offset)
            offset += episode["patients"]
        histories, labels = self.core.encode_support(memory["embeddings"][ids], torch.cat(owners),
                                                    memory["classes"][ids])
        return histories, labels

    @torch.no_grad()
    def _fit_missing(self, episodes) -> None:
        missing = [episode for episode in episodes if episode["key"] not in self._plans]
        if not missing:
            return
        modes = [(module, module.training) for module in self.core.modules()]
        try:
            self.core.eval()
            device = next(self.parameters()).device
            with torch.autocast(device.type, enabled=torch.is_autocast_enabled(device.type),
                                dtype=torch.get_autocast_dtype(device.type), cache_enabled=False):
                _, labels = self._encode_episodes(missing)
            # Native clustering is explicitly CPU/SciPy. Transfer one batched
            # teacher table, then fit independent CPU episode plans in parallel.
            label_parts = labels.detach().float().cpu().split([episode["patients"] for episode in missing])
            native_cluster = importlib.import_module("hiercp_v222.clustering")
            tasks = [(episode, label, episode["owners"].cpu(), episode["classes"].cpu())
                     for episode, label in zip(missing, label_parts)]
            def fit(task):
                episode, label, owners_cpu, classes_cpu = task
                plan = native_cluster.fit_prototypes(label, owners_cpu, classes_cpu)
                return episode["key"], plan
            try:
                cores = len(os.sched_getaffinity(0))
            except AttributeError:
                cores = os.cpu_count() or 1
            if len(tasks) > 1 and cores > 1:
                with ThreadPoolExecutor(max_workers=min(cores, len(tasks))) as pool:
                    fitted = list(pool.map(fit, tasks))
            else:
                fitted = [fit(task) for task in tasks]
            for key, plan in fitted:
                self._plans[key] = _to_device(plan, device)
        finally:
            for module, mode in modes:
                module.training = mode

    def _aligned_labels(self, episodes, labels):
        sizes = [episode["patients"] * 2 for episode in episodes]
        width = max(sizes)
        pieces = labels.flatten(0, 1).split(sizes)
        aligned = torch.stack([F.pad(piece, (0, 0, 0, width - len(piece))) for piece in pieces])
        length = torch.tensor(sizes, device=aligned.device)
        slots = torch.arange(width, device=aligned.device)
        padding = slots[None] >= length[:, None]
        task = slots // 2
        forbidden = task[:, None] == task[None, :]
        for layer, update in zip(self.core.l2, self.core.l2_updates):
            def run(value, layer=layer, update=update):
                message = layer(value, value, value, attn_mask=forbidden,
                                key_padding_mask=padding, need_weights=False)[0]
                return update(value, message)
            aligned = checkpoint(run, aligned, use_reentrant=False) if self.training and self.core.checkpoint_support else run(aligned)
        return aligned, sizes

    def score(self, query: Tensor, case_ids, counts) -> list[Tensor]:
        memory = self._require_memory()
        cases = _strings(case_ids, "actual query case")
        if (not isinstance(counts, (tuple, list)) or len(counts) != len(cases)
                or any(type(count) is not int or count < 1 for count in counts)
                or query.ndim != 2 or query.shape != (sum(counts), 128)
                or query.device != memory["embeddings"].device or not query.is_floating_point()):
            raise ValueError("Real original local fused128 embeddings and aligned complete query ownership required")
        if self.training and any(count != 8 for count in counts):
            raise ValueError("Original training must retain every eight-candidate curriculum sample")
        if not self.debug_support and query.device.type != "cuda":
            raise RuntimeError("Half-B production requires actual CUDA; no CPU neural fallback")
        allowed = set(memory["training_case_ids"]) if self.training else set(memory["training_case_ids"]) | set(memory["validation_case_ids"])
        if not set(cases) <= allowed:
            raise ValueError("Query case is outside the preserved train/validation cohort")
        if self.training and not set(cases) <= set(memory["patient_case_ids"]):
            raise ValueError("Actual training query is absent from the complete signed support cache")
        _require(torch.isfinite(query).all(), "Nonfinite original local query embeddings")
        keys = [case if case in memory["patient_case_ids"] else None for case in cases]
        unique = list(dict.fromkeys(keys))
        episodes = [self._episode(key) for key in unique]
        sample_episode = torch.tensor([unique.index(key) for key in keys], device=query.device)
        query_episode = torch.repeat_interleave(sample_episode, torch.tensor(counts, device=query.device))
        self._fit_missing(episodes)
        histories, local_labels = self._encode_episodes(episodes)
        aligned, sizes = self._aligned_labels(episodes, local_labels)
        width = max(sizes)
        q = query
        # All candidates/cases share batched tensor calls. An episode mask is
        # additional disjoint ownership, never a neighbor/sample reduction.
        for layer, history in zip(self.core.l1, histories):
            pieces = history.split(sizes)
            labels = torch.stack([F.pad(piece, (0, 0, 0, width - len(piece))) for piece in pieces])
            valid = torch.arange(width, device=q.device)[None] < torch.tensor(sizes, device=q.device)[:, None]
            label_ids = torch.arange(len(episodes) * width, device=q.device).reshape(len(episodes), width)
            selected = label_ids[query_episode]
            keep = valid[query_episode]
            src = selected[keep]
            dst = torch.arange(len(q), device=q.device)[:, None].expand_as(selected)[keep]
            q = layer.messages(labels.flatten(0, 1), q, src, dst, q.new_zeros(len(src), 2))
        centers, classes, priors, prototype_episodes = [], [], [], []
        native_cluster = importlib.import_module("hiercp_v222.clustering")
        for index, (episode, size) in enumerate(zip(episodes, sizes)):
            plan = self._plans[episode["key"]]
            # Prototype reductions have different discovered K; each uses all
            # actual memberships. The learned L1/L2/query operations above are
            # batched; these tiny deterministic ragged reductions add no cap.
            labels = aligned[index, :size].reshape(episode["patients"], 2, 128)
            centers.append(native_cluster.live_prototypes(labels, plan))
            classes.append(plan["prototype_classes"])
            priors.append(plan["log_prior"])
            prototype_episodes.append(torch.full_like(plan["prototype_classes"], index))
        with torch.autocast(query.device.type, enabled=False):
            cosine = F.normalize(q.float(), dim=-1) @ torch.cat(centers).T / self.core.temperature
            mixture = cosine + torch.cat(priors)
            valid_episode = query_episode[:, None] == torch.cat(prototype_episodes)[None]
            class_mask = torch.cat(classes)[:, None] == torch.arange(2, device=q.device)[None]
            logits = torch.logsumexp(mixture[:, :, None].masked_fill(
                ~(valid_episode[:, :, None] & class_mask[None]), -torch.inf), dim=1)
            score = logits[:, 1] - logits[:, 0]
        _require(torch.isfinite(score).all(), "Nonfinite native clustered cosine ranking scores")
        return list(score.split(tuple(counts)))


def _score_upper(self, batch, local_embeddings):
    if not isinstance(local_embeddings, dict) or "fused" not in local_embeddings:
        raise ValueError("Half B requires the genuine original v1 semantic pair readout")
    return self.half_b.score(local_embeddings["fused"], batch.case_ids, batch.counts)


def install_half_b(model: nn.Module, *, debug_support=False) -> nn.Module:
    """Replace upper ownership only, preserving original L0 bytes and RNG."""
    if hasattr(model, "half_b"):
        if model.half_b.debug_support != bool(debug_support):
            raise ValueError("Half-B DEBUG/production support profile cannot be silently changed")
        return model
    local = model.local_encoder
    if (getattr(model, "hidden_dim", None) != 128 or getattr(model, "ablation_mode", None) != "full"
            or getattr(local, "hidden_dim", None) != 128 or len(getattr(local, "project", {})) != 6
            or len(getattr(local, "blocks", ())) != 3 or local.dense_batch_size != 4
            or getattr(model, "v1x_stage", None) is not None or hasattr(model, "half_a_mode")):
        raise ValueError("Construct a fresh complete ORIGINAL v1 L0128D/4heads/3layers for half B")
    for name in ("patient_encoder", "prototype_encoder"):
        blocks = getattr(model, name).blocks
        if len(blocks) != 2 or any(conv.heads != 4 for block in blocks for conv in block.conv.convs.values()):
            raise ValueError("Original128D/four-head/two+two upper constructor required before replacement")
    scope = re.search(r"\|bounded_scope_([0-9a-f]{64})(?:\||$)", model.architecture_version)
    if scope is None:
        raise ValueError("Half B requires the actual preserved original10mm bounded scope identity")
    parameter = next(local.parameters())
    # The new constructor uses only the CPU generator; .to() draws no random
    # numbers. Never reseed CUDA from a CPU constructor after CUDA is initialized.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        with torch.device("cpu"):
            upper = HalfBUpper(dropout=local.final_fuse[3].p, scope_contract=scope.group(1),
                               debug_support=debug_support)
        upper.to(device=parameter.device, dtype=parameter.dtype)
    upper.train(model.training)
    for name in _REMOVED_UPPER:
        delattr(model, name)
    model.half_b = upper
    model._score_upper = MethodType(_score_upper, model)
    model.architecture_version += ARCHITECTURE_SUFFIX
    model.half_b_mode = MODE
    model.register_buffer(MARKER_KEY, checkpoint_marker(debug_support=debug_support).to(parameter.device))
    model.register_load_state_dict_pre_hook(_reject_other_half)
    return model
