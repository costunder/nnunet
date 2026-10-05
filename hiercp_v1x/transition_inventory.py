"""Complete, metadata-only v1-m10 -> native LocalCNN v2.2 transition inventory.

The old half-A/B runs changed architecture while retaining the v1 task. They
are evidence about those bridges, not complete halves of the native transition.
This module never constructs a model, executes training or changes an existing
experiment. A partition is a design; it is not an executable implementation.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


FORMAT = "hiercp_complete_v1_m10_to_localcnn_v22_transition_v1"
PLAN_FORMAT = "hiercp_complete_transition_partition_v1"
SMOKE_FORMAT = "hiercp_complete_transition_cuda_smoke_v1"
BASELINE_TASK = "v1_source_original_anchor_vs_curriculum_candidates"
TARGET_TASK = "v22_observed_eligible_tumor_P_vs_unobserved_U"
PINNED_V1_ARCHIVE_SHA256 = "5157bafe641e9189824826532a3b055ea560f5c5dfc299374842a3bd1d20a22e"
SOURCE_FILES = (
    "hiercp_v1x/transition_inventory.py", "versions/v1/pipeline_v1_source.zip", "hiercp_v1x/bounded_scope.py",
    "hiercp_v1x/scope_training_entry.py", "hiercp_v1x/half_a_model.py",
    "hiercp_v1x/half_b_model.py", "hiercp_v1x/half_b_support.py",
    "l0_local_cnn/data.py", "l0_local_cnn/model.py", "l0_exploration/model.py",
    "l0_regions/donor_data.py", "l0_regions/donor_learning.py",
    "l0_regions/training.py", "l0_regions/support_episodes.py",
    "hiercp_v222/model.py", "hiercp_v222/clustering.py",
    "tools/v22_rank_objective.py", "tools/v22_candidate_order.py",
    "tools/run_local_cnn.py", "config/prompt_graph_v222_v1_l0.json",
)


def _factor(name, domain, baseline, target, sources, a, b):
    return dict(name=name, domain=domain, baseline=baseline, target=target,
                source_files=list(sources), existing_half_A=a, existing_half_B=b)


# "covered" is limited to the named atomic operation, never native end-to-end
# equivalence. "partial" explicitly includes bridges with preserved v1 inputs.
_FACTORS = (
    _factor("donor_condition", "input", "source tumor and its own original anchor",
            "independent inner-train donor fixed per recipient; P/U labels independent of donor",
            ("l0_regions/donor_data.py",), "not_covered", "not_covered"),
    _factor("physical_input_sampling", "input", "adaptive physical ROI resampled to fixed 48 cubed",
            "native-spacing variable crops; no image resizing",
            ("l0_local_cnn/data.py",), "not_covered", "not_covered"),
    _factor("input_channels", "input", "five-channel v1 payload including geometric masks",
            "one CT channel masked by liver; separate organ validity mask",
            ("l0_local_cnn/data.py", "hiercp_v1x/half_a_model.py"), "partial", "not_covered"),
    _factor("recipient_tumor_erasure", "input", "original v1 target erasure retained",
            "native observed CT crop; no v1 target-erasure operation",
            ("l0_local_cnn/data.py", "hiercp_v1x/half_a_model.py"), "not_covered", "not_covered"),
    _factor("role_shell_geometry", "input", "six local roles, physical shells and handcrafted node features",
            "no sampled roles, shells, handcrafted features or local graph",
            ("l0_local_cnn/model.py", "hiercp_v1x/half_a_model.py"), "partial", "not_covered"),
    _factor("local_encoder", "L0", "original dense v1 encoder",
            "OrganPyramid channels 12/24/32; convolutions 2/3/3",
            ("l0_exploration/model.py", "hiercp_v1x/half_a_model.py"), "covered", "not_covered"),
    _factor("local_message_passing", "L0", "three-layer heterogeneous local GAT",
            "no local graph message passing; CNN maps supply features",
            ("l0_local_cnn/model.py", "hiercp_v1x/half_a_model.py"), "covered", "not_covered"),
    _factor("local_readout", "L0", "role/shell attention readouts plus handcrafted context",
            "organ masked mean per pyramid scale then 68-to-128 projection",
            ("l0_local_cnn/model.py", "hiercp_v1x/half_a_model.py"), "partial", "not_covered"),
    _factor("pair_fusion", "L0", "original multiple semantic pair and fused outputs",
            "one donor/recipient difference-product fusion to 128D",
            ("l0_local_cnn/model.py", "hiercp_v1x/half_a_model.py"), "partial", "not_covered"),
    _factor("upper_L1", "upper", "v1 patient graph semantics and heterogeneous GAT",
            "tracked legacy prompt RelationLayer twice; not official PRODIGY reference",
            ("hiercp_v222/model.py", "hiercp_v1x/half_b_model.py"), "not_covered", "covered"),
    _factor("upper_L2", "upper", "v1 region/prototype graph and population encoder",
            "tracked two cross-patient alignment layers and class-conditional clusters",
            ("hiercp_v222/model.py", "hiercp_v222/clustering.py"), "not_covered", "covered"),
    _factor("scalar_score", "upper", "direct v1 scalar head",
            "patient-mass cosine mixture logit1 minus logit0",
            ("hiercp_v222/model.py", "hiercp_v1x/half_b_model.py"), "not_covered", "covered"),
    _factor("GT_semantics", "task", BASELINE_TASK, TARGET_TASK,
            ("l0_regions/donor_learning.py", "config/prompt_graph_v222_v1_l0.json"), "not_covered", "not_covered"),
    _factor("training_sample_population", "task", "selected source samples; original samples_per_case=2",
            "all retained observed P and 128 U per recipient, including cases without P",
            ("l0_local_cnn/data.py", "l0_regions/donor_learning.py"), "not_covered", "not_covered"),
    _factor("training_candidate_set", "task", "anchor plus seven curriculum comparisons from target pool 128",
            "every P times all 128 U; both comparison sides encoded live",
            ("l0_regions/donor_learning.py", "l0_local_cnn/data.py"), "not_covered", "not_covered"),
    _factor("comparison_corruption", "task", "original relation-corrupted curriculum negatives",
            "original unobserved centers without v1 relation corruption",
            ("versions/v1/pipeline_v1_source.zip", "l0_local_cnn/data.py"), "not_covered", "not_covered"),
    _factor("ranking_loss", "objective", "listwise, difficulty-margin pairwise, ordinal and mined objectives",
            "global all-P/U softplus pair mean with explicit tile normalization",
            ("versions/v1/pipeline_v1_source.zip", "l0_regions/donor_learning.py"), "not_covered", "not_covered"),
    _factor("observation_alignment_auxiliaries", "objective", "no native observation/alignment CE",
            "balanced observation CE inverse presentation multiplicity plus native alignment CE",
            ("l0_regions/donor_learning.py", "hiercp_v222/model.py"), "not_covered", "not_covered"),
    _factor("two_view_consistency", "objective", "original two graph views and six-term consistency",
            "native LocalCNN single crop-pair representation without v1 consistency",
            ("l0_local_cnn/model.py", "l0_regions/donor_learning.py"), "not_covered", "not_covered"),
    _factor("support_labels_population", "support", "no native observation support; original v1 upper context",
            "training-only observed-P/U detached L0 memory, exclude recipient and donor patient groups",
            ("hiercp_v1x/half_b_support.py", "l0_regions/training.py"), "not_covered", "partial"),
    _factor("support_refresh", "support", "original v1 patient/region prototype preparation",
            "all training L0 memory initially and each epoch; detached teacher cluster plans",
            ("l0_regions/training.py", "hiercp_v1x/half_b_model.py"), "not_covered", "partial"),
    _factor("support_episode_selection", "support", "original v1 context; B bridge uses all eligible anchor/curriculum support",
            "explicit native support_patients policy during optimization; full support evaluation",
            ("l0_regions/support_episodes.py", "l0_regions/training.py"), "not_covered", "not_covered"),
    _factor("update_schedule", "schedule", "v1 difficulty curriculum, eight-candidate samples and two views",
            "complete case-wise P times U live tiles; repeated CE compensated; alignment follows native tile weighting",
            ("l0_regions/donor_learning.py", "l0_regions/training.py"), "not_covered", "not_covered"),
    _factor("optimizer_LR_schedule", "schedule", "original AdamW plus CosineAnnealingLR",
            "native AdamW with constant configured LR; no original cosine scheduler",
            ("versions/v1/pipeline_v1_source.zip", "l0_regions/training.py"), "not_covered", "not_covered"),
    _factor("precision_and_scaler", "execution", "original AMP autocast plus GradScaler",
            "native FP32 optimization without original autocast/scaler path",
            ("versions/v1/pipeline_v1_source.zip", "l0_regions/training.py"), "not_covered", "not_covered"),
    _factor("physical_batch_unit_and_work", "execution", "measured v1 cache-sample batch; each sample includes eight candidates and two views",
            "measured native observation-row batch and live P/U tiling; explicit actual candidate work",
            ("l0_regions/donor_learning.py", "l0_regions/training.py"), "not_covered", "not_covered"),
    _factor("activation_storage", "execution", "original CNN/local graph activation checkpointing and captured RNG policy",
            "native explicit retained/checkpointed policy from actual run receipt; never infer from hardware capacity",
            ("l0_regions/training.py", "tools/run_local_cnn.py"), "not_covered", "not_covered"),
    _factor("evaluation_candidate_universe", "evaluation", "fixed full-difficulty eight-candidate v1 validation",
            "each held-out recipient's all P plus 128 U scored once without corruption",
            ("l0_regions/training.py", "tools/v22_rank_objective.py"), "not_covered", "not_covered"),
    _factor("checkpoint_selection", "evaluation", "original v1 MRR/top1/margin tie-breaking",
            "native same_donor_live_v1 MRR then R@1 then negative pairwise loss",
            ("l0_regions/donor_learning.py",), "not_covered", "not_covered"),
)
FACTOR_NAMES = tuple(f["name"] for f in _FACTORS)

# Controls are never silently copied from hardware auto-calibration. The exact
# physical batch unit matters: one v1 sample includes eight candidate graphs,
# whereas one native query row denotes one observation. Numeric equality alone
# is not equality of simultaneous candidate work.
CONTROL_FIELDS = (
    "seed", "epochs", "margin_mm", "split_sha256", "raw_cohort_sha256",
    "outer_test_policy", "initialization_policy", "optimizer_policy",
    "physical_batch_policy", "precision_policy", "runtime_resource_policy",
)


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} requires a lowercase SHA256")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def inventory():
    return dict(format=FORMAT, baseline="completed_v1_m10", target="native_localcnn_v22_same_donor_live_v1",
                baseline_task=BASELINE_TASK, target_task=TARGET_TASK,
                target_semantics=dict(P="actual observed eligible tumor anchor", U="unobserved comparison center",
                                      P_is_donor_compatibility_GT=False, U_is_CP_ineligible_GT=False),
                factors=copy.deepcopy(list(_FACTORS)), source_files=list(SOURCE_FILES),
                scope="complete transition design inventory; existing A/B are architecture bridges",
                runnable=False, quality_verified=False, training_started=False)


def validate_inventory(value):
    if not isinstance(value, Mapping) or dict(value) != inventory():
        raise ValueError("Inventory differs from complete declared source/task/coverage contract")
    return copy.deepcopy(dict(value))


def existing_coverage():
    full = [f["name"] for f in _FACTORS if "covered" in (f["existing_half_A"], f["existing_half_B"])]
    partial = [f["name"] for f in _FACTORS if f["name"] not in full and "partial" in (f["existing_half_A"], f["existing_half_B"])]
    missing = [name for name in FACTOR_NAMES if name not in full and name not in partial]
    return dict(fully_covered=full, partially_covered=partial, not_covered=missing,
                all_transition_changes_covered=False, native_endpoint_equivalence=False)


def validate_coverage_claim(claim):
    if not isinstance(claim, Mapping) or dict(claim) != existing_coverage():
        raise ValueError("A/B cannot be relabeled as a complete native-v2.2 transition")
    return copy.deepcopy(dict(claim))


def _names(values, label):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"Explicit nonempty {label} factor list required")
    if any(not isinstance(v, str) for v in values) or len(set(values)) != len(values):
        raise ValueError(f"Duplicate or invalid {label} factors")
    if set(values) - set(FACTOR_NAMES):
        raise ValueError(f"Unknown {label} factors: {sorted(set(values) - set(FACTOR_NAMES))}")
    return [name for name in FACTOR_NAMES if name in values]


def validate_partition(groups):
    if not isinstance(groups, Mapping) or set(groups) != {"half_A", "half_B"}:
        raise ValueError("Exactly two explicitly named complete-transition halves required")
    normalized = {name: _names(groups[name], name) for name in ("half_A", "half_B")}
    counts = Counter(v for values in normalized.values() for v in values)
    missing = set(FACTOR_NAMES) - set(counts)
    duplicate = {name for name, count in counts.items() if count != 1}
    if missing or duplicate:
        raise ValueError(f"Every transition factor must occur exactly once; missing={sorted(missing)}, repeated={sorted(duplicate)}")
    return normalized


def validate_controls(controls):
    if not isinstance(controls, Mapping) or set(controls) != set(CONTROL_FIELDS):
        raise ValueError("All explicit controlled cohort, scope, seed and runtime policies required")
    result = copy.deepcopy(dict(controls))
    if type(result["seed"]) is not int or result["seed"] != 42 or type(result["epochs"]) is not int or result["epochs"] != 40:
        raise ValueError("This transition preserves seed42 and 40 final epochs; DEBUG is separate")
    if isinstance(result["margin_mm"], bool) or not isinstance(result["margin_mm"], (int, float)) or not math.isfinite(result["margin_mm"]) or result["margin_mm"] != 10:
        raise ValueError("This transition preserves the completed baseline's 10mm scope")
    for key in ("split_sha256", "raw_cohort_sha256"):
        _hash(result[key], key)
    if result["outer_test_policy"] != "untouched":
        raise ValueError("Outer test remains untouched during localization")
    for key in CONTROL_FIELDS[6:]:
        if not isinstance(result[key], Mapping) or not result[key]:
            raise ValueError(f"Explicit {key} policy object required")
    if "unit" not in result["physical_batch_policy"] or "candidate_work" not in result["physical_batch_policy"]:
        raise ValueError("Physical batch unit and simultaneous candidate work must be explicit")
    digest(result)
    return result


def make_plan(groups, *, controls):
    groups = validate_partition(groups)
    controls = validate_controls(controls)
    value = dict(format=PLAN_FORMAT, inventory_sha256=digest(inventory()), groups=groups, controls=controls,
                 baseline_task=BASELINE_TASK, target_task=TARGET_TASK, same_GT=False,
                 native_endpoint_equivalence=False,
                 task_change="declared observation task transition; never donor compatibility relabeling",
                 interaction_assumption="not monotonic; both individual halves may pass while their interaction fails",
                 evaluation_policy="report own-task metrics separately; use one bound common observation audit without relabeling v1 training GT",
                 implementation_registration={}, runnable={"half_A": False, "half_B": False},
                 quality_verified=False, training_started=False)
    value["contract_sha256"] = digest(value)
    return value


def validate_plan(plan):
    if not isinstance(plan, Mapping):
        raise ValueError("Explicit complete transition plan required")
    expected = make_plan(plan.get("groups"), controls=plan.get("controls"))
    if dict(plan) != expected:
        raise ValueError("Unregistered plan changed, claims same GT or silently became runnable")
    return expected


def source_bindings(root):
    """Capture real source bytes; this does not register an executable bridge."""
    root = Path(root).resolve(strict=True)
    result = {}
    for name in SOURCE_FILES:
        path = root.joinpath(*PurePosixPath(name).parts)
        data = path.read_bytes()
        result[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if name == "versions/v1/pipeline_v1_source.zip" and result[name]["sha256"] != PINNED_V1_ARCHIVE_SHA256:
            raise ValueError("Original v1 source archive differs from the pinned baseline")
    return result


def validate_source_bindings(root, bindings):
    if not isinstance(bindings, Mapping) or dict(bindings) != source_bindings(root):
        raise ValueError("Transition implementation source bytes or inventory changed")
    return copy.deepcopy(dict(bindings))


def register_cuda_smoke(plan, group, *, root, recipe, receipt, artifact):
    """Return separate mechanical readiness only after a bound real-CT receipt.

    The caller supplies a real implementation recipe and on-disk smoke result;
    no synthetic receipt or standalone source hash is sufficient. Registration
    is in memory only and never marks quality or full evaluation as complete.
    """
    plan = validate_plan(plan)
    if group not in plan["groups"]:
        raise ValueError("Unknown complete-transition half")
    if not isinstance(recipe, Mapping) or set(recipe) != {"factors", "source_bindings", "bridge_sources", "controls_sha256", "runner", "task_contract", "debug_profile_separate"}:
        raise ValueError("Complete explicitly implemented recipe required")
    selected = plan["groups"][group]
    expected = {f["name"]: (f["target"] if f["name"] in selected else f["baseline"]) for f in _FACTORS}
    if recipe["factors"] != expected or recipe["controls_sha256"] != digest(plan["controls"]) or recipe["debug_profile_separate"] is not True:
        raise ValueError("Runtime recipe omitted a transition factor or changed a control")
    validate_source_bindings(root, recipe["source_bindings"])
    bridge_sources = recipe["bridge_sources"]
    if not isinstance(bridge_sources, Mapping) or not bridge_sources:
        raise ValueError("Actual bridge/runner source hashes required")
    for name, source_hash in bridge_sources.items():
        parts = PurePosixPath(name).parts
        if not parts or PurePosixPath(name).is_absolute() or ".." in parts or "\\" in name or ":" in name:
            raise ValueError("Bridge source must be a safe repository-relative path")
        _hash(source_hash, name)
        if hashlib.sha256(Path(root).joinpath(*parts).read_bytes()).hexdigest() != source_hash:
            raise ValueError("Bridge source changed after smoke")
    if recipe["runner"] not in bridge_sources:
        raise ValueError("Actual executable runner must be bound among bridge sources")
    task = TARGET_TASK if "GT_semantics" in selected else BASELINE_TASK
    if recipe["task_contract"] != task:
        raise ValueError("Runtime task does not match declared GT transition")
    artifact = Path(artifact).resolve(strict=True)
    artifact_bytes = artifact.read_bytes()
    stored = json.loads(artifact_bytes.decode("utf-8"))
    if not isinstance(receipt, Mapping) or dict(receipt) != stored:
        raise ValueError("Supplied CUDA evidence differs from its saved artifact")
    required = {"format", "plan_sha256", "recipe_sha256", "group", "device", "scope", "input_case_ids", "input_record_ids", "factor_audits", "forward", "loss", "backward", "optimizer", "evaluation", "query_GT_in_forward", "resources", "quality_verified", "full_training_complete"}
    if set(stored) != required or stored["format"] != SMOKE_FORMAT:
        raise ValueError("Complete transition CUDA smoke receipt required")
    if stored["plan_sha256"] != plan["contract_sha256"] or stored["recipe_sha256"] != digest(recipe) or stored["group"] != group:
        raise ValueError("Smoke was run for a different plan/recipe/half")
    if not isinstance(stored["device"], str) or not stored["device"].startswith("cuda:") or stored["scope"] != "DEBUG_real_CT_mechanical_only":
        raise ValueError("Real-CT CUDA mechanical smoke required; no CPU or dummy promotion")
    for key in ("input_case_ids", "input_record_ids"):
        values = stored[key]
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values) or len(values) != len(set(values)):
            raise ValueError("Actual unique real-CT case/record identities required")
    if stored["factor_audits"] != expected or any(stored[k] is not True for k in ("forward", "loss", "backward", "optimizer", "evaluation")):
        raise ValueError("Every actual factor and complete neural path must be audited")
    if stored["query_GT_in_forward"] is not False or stored["quality_verified"] is not False or stored["full_training_complete"] is not False:
        raise ValueError("Smoke cannot leak query GT or claim final quality/training")
    resources = stored["resources"]
    if not isinstance(resources, Mapping) or set(resources) != {"gpu", "physical_batch", "physical_batch_unit", "candidate_work", "peak_cuda_bytes", "peak_rss_bytes", "update_seconds", "workers"}:
        raise ValueError("Measured batch, actual candidate work, GPU/RAM/time and workers required")
    if not isinstance(resources["gpu"], str) or not resources["gpu"] or not isinstance(resources["physical_batch_unit"], str) or not resources["physical_batch_unit"]:
        raise ValueError("Actual hardware and physical-batch unit required")
    for key in ("physical_batch", "candidate_work", "peak_cuda_bytes", "peak_rss_bytes"):
        if type(resources[key]) is not int or resources[key] <= 0:
            raise ValueError("Positive measured resource fields required")
    if type(resources["workers"]) is not int or resources["workers"] < 0 or isinstance(resources["update_seconds"], bool) or not isinstance(resources["update_seconds"], (int, float)) or not math.isfinite(resources["update_seconds"]) or resources["update_seconds"] <= 0:
        raise ValueError("Actual finite update cost and worker count required")
    return dict(plan_sha256=plan["contract_sha256"], group=group, recipe_sha256=digest(recipe),
                source_sha256=digest(recipe["source_bindings"]), artifact_path=str(artifact),
                artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
                mechanical_cuda_smoke_verified=True, runnable=True,
                native_endpoint_equivalence=False, quality_verified=False, full_training_complete=False)
