"""Implementation and patient-partition contracts (no tensor dependencies)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

GEOMETRY_CONTRACT = "level0_physical_closure_v2"
ARCHITECTURE_VERSION = "hiercp_source_content_population_metric_v5"
PATIENT_GRAPH_CONTRACT = "patient_source_content_population_v2"


def require_current_patient_graph(payload: Mapping) -> None:
    """Validate persisted upper-graph semantics before applying recipe defaults.

    L0 physical geometry is unchanged. The v5 upper graph additionally binds
    observed-CT region descriptors and typed population-distance edges. Missing
    or old metadata never permits reinterpreting earlier learned assets.
    """
    if (not isinstance(payload, Mapping)
            or payload.get("patient_graph_contract") != PATIENT_GRAPH_CONTRACT):
        raise ValueError(
            "Missing or incompatible patient_graph_contract; preserve the old "
            "source-host graph and rebuild upper graphs in a NEW workspace."
        )


def require_current_checkpoint(payload: Mapping) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping")
    expected = {"architecture_version": ARCHITECTURE_VERSION,
                "geometry_contract": GEOMETRY_CONTRACT}
    wrong = [name for name, value in expected.items() if payload.get(name) != value]
    graph_config = payload.get("graph_config")
    if not isinstance(graph_config, Mapping):
        graph_config = {}
    if graph_config.get("geometry_contract") != GEOMETRY_CONTRACT:
        wrong.append("graph_config.geometry_contract")
    if graph_config.get("patient_graph_contract") != PATIENT_GRAPH_CONTRACT:
        wrong.append("graph_config.patient_graph_contract")
    if wrong:
        raise ValueError(
            "Checkpoint belongs to an older/incompatible experiment: " + ", ".join(wrong)
            + ". Preserve it for legacy evaluation; rebuild graphs and retrain in a NEW workspace."
        )


def case_ids(values: Sequence, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(v, str) or not v.strip() for v in values
    ):
        raise ValueError(f"{name} must be a list of nonempty patient IDs")
    result = list(values)
    if (not result and not allow_empty) or len(result) != len(set(result)):
        raise ValueError(f"{name} is empty or contains duplicate patient IDs")
    return result


def validate_nested_cohorts(train, validation, inner_train, inner_validation, prototypes):
    """Test labels may only enter downstream evaluation, never any GNN stage."""
    train = case_ids(train, "outer train")
    validation = case_ids(validation, "outer validation")
    inner_train = case_ids(inner_train, "GNN train")
    inner_validation = case_ids(inner_validation, "GNN validation")
    prototypes = case_ids(prototypes, "prototype fit")
    if set(train) & set(validation):
        raise ValueError("Outer train/validation patient leakage")
    if set(inner_train) & set(inner_validation):
        raise ValueError("GNN train/validation patient leakage")
    if set(inner_train) | set(inner_validation) != set(train):
        raise ValueError("GNN cohorts must exactly partition the outer training patients")
    if set(prototypes) != set(inner_train):
        raise ValueError("Population prototypes must be fitted only to GNN training patients")
