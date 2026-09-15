"""Cross-patient population prototype bank for liver-region descriptors."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
import torch

from hiercp.region import REGION_DESCRIPTOR_POLICY, numpy_kmeans
from hiercp.schema import PROTOTYPE_FEATURE_DIM, REGION_FEATURE_DIM, GraphBuildConfig


PROTOTYPE_BANK_FORMAT = "hiercp_prototype_bank_v2"
PROTOTYPE_FIT_CONTRACT = "numpy_kmeans_final_nearest_membership_v2"


def _load_bank_payload(path: str | Path) -> dict:
    """Read tensor/primitive-only banks without executing arbitrary pickle code.

    Both the historical v1 writer and the current writer use this restricted
    payload vocabulary. Unsupported globals and older PyTorch APIs fail closed;
    auditing an old bank is never permission to retry with unsafe pickle loading.
    """
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Prototype banks require a PyTorch version supporting weights_only=True; "
            "unsafe pickle fallback is not permitted"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Prototype bank payload must be a dictionary")
    return payload


def _array_fingerprint(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _json_value(value):
    """Record NumPy RNG provenance without executable/pickled custom objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _payload_array(payload: dict, name: str, dtype) -> np.ndarray:
    value = payload[name]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise ValueError(f"Prototype payload {name} must have dtype {np.dtype(dtype)}")
    return value.copy()


@dataclass
class PrototypeBank:
    """Population prototypes fitted only on the training-case region graphs."""

    features: np.ndarray
    standardized_centers: np.ndarray
    descriptor_mean: np.ndarray
    descriptor_std: np.ndarray
    edge_index: np.ndarray
    training_case_ids: tuple[str, ...]
    # In-memory synthetic/custom banks may omit fit evidence, but they cannot be
    # serialized as current fitted banks. Current file loading always requires it.
    fit_descriptors: np.ndarray | None = None
    fit_labels: np.ndarray | None = None
    fit_counts: np.ndarray | None = None
    fit_case_counts: np.ndarray | None = None
    fit_provenance: dict | None = None

    @property
    def num_prototypes(self) -> int:
        return int(self.features.shape[0])

    def fingerprint(self) -> str:
        """Content hash used to bind caches/checkpoints to this exact bank."""

        digest = hashlib.sha256()
        digest.update(PROTOTYPE_BANK_FORMAT.encode("utf-8"))
        digest.update(PROTOTYPE_FIT_CONTRACT.encode("utf-8"))
        digest.update(REGION_DESCRIPTOR_POLICY.encode("utf-8"))
        for array in (
            self.features,
            self.standardized_centers,
            self.descriptor_mean,
            self.descriptor_std,
            self.edge_index,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("utf-8"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        for case_id in self.training_case_ids:
            digest.update(case_id.encode("utf-8"))
            digest.update(b"\0")
        for value in (self.fit_descriptors, self.fit_labels, self.fit_counts, self.fit_case_counts):
            digest.update(b"absent" if value is None else _array_fingerprint(value).encode("ascii"))
        digest.update(json.dumps(self.fit_provenance, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))
        return digest.hexdigest()

    def validate(self, *, verify_fit: bool = False) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != PROTOTYPE_FEATURE_DIM:
            raise ValueError(f"Invalid prototype features: {self.features.shape}")
        if self.standardized_centers.shape != (self.num_prototypes, REGION_FEATURE_DIM):
            raise ValueError("Invalid standardized prototype centers")
        if self.descriptor_mean.shape != (REGION_FEATURE_DIM,):
            raise ValueError("Invalid descriptor mean")
        if self.descriptor_std.shape != (REGION_FEATURE_DIM,):
            raise ValueError("Invalid descriptor std")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.num_prototypes < 1:
            raise ValueError("A prototype bank must contain every configured prototype")
        if any(not np.isfinite(value).all() for value in (
            self.features, self.standardized_centers, self.descriptor_mean, self.descriptor_std
        )) or np.any(self.descriptor_std <= 0):
            raise ValueError("Prototype descriptors/scaler must be finite with positive standard deviations")
        if not np.issubdtype(self.edge_index.dtype, np.integer) or (
            self.edge_index.size and (self.edge_index.min() < 0 or self.edge_index.max() >= self.num_prototypes)
        ):
            raise ValueError("Prototype edge indices are invalid")
        if verify_fit:
            self._validate_fit()

    @property
    def cluster_mean_distance(self) -> np.ndarray:
        """Mean within-cluster standardized Euclidean distance, not confidence."""
        return self.features[:, REGION_FEATURE_DIM + 1]

    def standardized_distances(self, region_features: np.ndarray) -> np.ndarray:
        """Return [regions, prototypes] absolute Euclidean distances in 16D.

        This is not the spatial XYZ distance or a calibrated outlier probability.
        A signed excess uses this distance minus ``cluster_mean_distance``; no
        epsilon division, clipping, or implicit rejection threshold is applied.
        """
        return np.sqrt(self._standardized_squared_distances(region_features))

    def _standardized_squared_distances(self, region_features: np.ndarray) -> np.ndarray:
        """One distance definition for fitting checks, assignment, and metrics."""
        self.validate()
        values = np.asarray(region_features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != REGION_FEATURE_DIM or not np.isfinite(values).all():
            raise ValueError("Region descriptors must be a finite [N, REGION_FEATURE_DIM] matrix")
        standardized = (values - self.descriptor_mean[None]) / self.descriptor_std[None]
        distances = np.sum((standardized[:, None] - self.standardized_centers[None]) ** 2, axis=-1)
        if not np.isfinite(distances).all():
            raise ValueError("Standardized prototype distances are non-finite")
        return distances.astype(np.float32, copy=False)

    def _validate_fit(self) -> None:
        """Recheck all saved fitting rows once at build/load/save boundaries.

        Centers are the last configured Lloyd M-step; final nearest membership
        need not be a stationary solution at the iteration budget. We validate
        its actual support/mean-distance, not an unjustified stationarity claim.
        """
        arrays = (self.fit_descriptors, self.fit_labels, self.fit_counts, self.fit_case_counts)
        if any(value is None for value in arrays) or not isinstance(self.fit_provenance, dict):
            raise ValueError("Current prototype serialization requires complete fit evidence")
        values, labels, counts, case_counts = arrays
        for value in (values, self.features, self.standardized_centers, self.descriptor_mean, self.descriptor_std):
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                raise ValueError("Fitted prototype numerical arrays must retain float32 dtype")
        for value in (labels, counts, case_counts, self.edge_index):
            if not isinstance(value, np.ndarray) or value.dtype != np.int64:
                raise ValueError("Fitted prototype index/count arrays must retain int64 dtype")
        if values.ndim != 2 or values.shape[1] != REGION_FEATURE_DIM or not len(values) or not np.isfinite(values).all():
            raise ValueError("Invalid full training fit descriptors")
        if labels.shape != (len(values),) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Invalid final fit labels")
        if counts.shape != (self.num_prototypes,) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("Invalid fit cluster counts")
        ids = self.training_case_ids
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(value, str) or not value or value != value.strip() for value in ids):
            raise ValueError("Fit training case IDs must be nonempty and unique")
        if case_counts.shape != (len(ids),) or not np.issubdtype(case_counts.dtype, np.integer) or np.any(case_counts <= 0) or case_counts.sum() != len(values):
            raise ValueError("Fit case descriptor counts do not cover the complete training matrix")
        proof = self.fit_provenance
        if proof.get("fit_contract") != PROTOTYPE_FIT_CONTRACT or proof.get("descriptor_policy") != REGION_DESCRIPTOR_POLICY:
            raise ValueError("Unsupported prototype fit/descriptor policy; refit into a new bank namespace")
        if proof.get("assignment_tie_policy") != "lowest_prototype_index":
            raise ValueError("Missing explicit prototype assignment tie policy")
        for name in ("max_iterations", "iterations_used", "clusters", "prototype_k", "final_reassigned_count"):
            if type(proof.get(name)) is not int:
                raise ValueError(f"Prototype fit provenance {name} must be an integer")
        if not 1 <= proof["iterations_used"] <= proof["max_iterations"] or proof["clusters"] != self.num_prototypes:
            raise ValueError("Invalid recorded prototype fit budget/count")
        if not 0 <= proof["final_reassigned_count"] <= len(values) or type(proof.get("membership_stable")) is not bool or proof["membership_stable"] != (proof["final_reassigned_count"] == 0):
            raise ValueError("Invalid terminal membership diagnostics")
        if proof.get("stopping_reason") not in ("iteration_budget", "center_allclose") or (proof["stopping_reason"] == "iteration_budget" and proof["iterations_used"] != proof["max_iterations"]):
            raise ValueError("Invalid prototype stopping reason")
        if not isinstance(proof.get("rng_initial_state"), dict) or not proof["rng_initial_state"].get("bit_generator") or "state" not in proof["rng_initial_state"] or not isinstance(proof.get("numpy_version"), str) or not proof["numpy_version"]:
            raise ValueError("Missing prototype numerical fitting provenance")
        json.dumps(proof, allow_nan=False, sort_keys=True)
        if proof.get("input_descriptor_sha256") != _array_fingerprint(values):
            raise ValueError("Prototype fit descriptor content fingerprint mismatch")
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
        if not np.array_equal(mean, self.descriptor_mean) or not np.array_equal(std, self.descriptor_std):
            raise ValueError("Prototype scaler is inconsistent with the full fit descriptors")
        if not np.allclose(self.features[:, :REGION_FEATURE_DIM], self.standardized_centers * std[None] + mean[None], rtol=1e-6, atol=1e-6):
            raise ValueError("Raw and standardized prototype centers disagree")
        squared = self._standardized_squared_distances(values)
        # sqrt can collapse adjacent float32 distances to a tie. Membership is
        # defined by the original squared distances, exactly as in k-means.
        nearest = squared.argmin(axis=1)
        distances = np.sqrt(squared)
        wanted_counts = np.bincount(nearest, minlength=self.num_prototypes)
        if not np.array_equal(labels, nearest) or not np.array_equal(counts, wanted_counts) or np.any(counts <= 0):
            raise ValueError("Final fit membership/counts disagree with saved centers")
        if not np.allclose(self.features[:, REGION_FEATURE_DIM], counts / len(values), rtol=1e-6, atol=1e-7):
            raise ValueError("Prototype support disagrees with final membership")
        dispersion = np.asarray([distances[nearest == cluster, cluster].mean() for cluster in range(self.num_prototypes)], dtype=np.float32)
        if not np.allclose(self.cluster_mean_distance, dispersion, rtol=1e-6, atol=1e-6):
            raise ValueError("Prototype dispersion disagrees with final membership")
        if not np.array_equal(self.edge_index, _prototype_knn(self.standardized_centers, proof["prototype_k"])):
            raise ValueError("Prototype KNN graph disagrees with saved standardized centers")

    def assign(
        self,
        region_features: np.ndarray,
        *,
        top_k: int,
        temperature: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        squared = self._standardized_squared_distances(region_features)
        if isinstance(top_k, (bool, np.bool_)) or int(top_k) != top_k:
            raise ValueError(f"top_k must be an exact integer, got {top_k!r}")
        k = int(top_k)
        if not 1 <= k <= self.num_prototypes:
            raise ValueError(
                "top_k must satisfy 1 <= top_k <= num_prototypes; "
                f"top_k={k}, num_prototypes={self.num_prototypes}"
            )
        temperature_value = float(temperature)
        if not np.isfinite(temperature_value) or temperature_value <= 0.0:
            raise ValueError(
                f"temperature must be finite and positive, got {temperature!r}"
            )
        # Stable ordering makes exact-distance ties agree with final argmin
        # membership and is deterministic across top-k requests.
        indices = np.argsort(squared, axis=1, kind="stable")[:, :k]
        ordered = np.take_along_axis(squared, indices, axis=1)
        logits = -ordered / temperature_value
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        normalizer = weights.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
            raise FloatingPointError("Prototype assignment produced invalid weights")
        weights /= normalizer
        return indices.astype(np.int64), weights.astype(np.float32)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Publish without clobbering history; identical current banks are reusable.

        ``overwrite`` is retained for caller compatibility, not permission to
        replace historical artifacts. A different bank requires a fresh path.
        """
        self.validate(verify_fit=True)
        destination = Path(path)
        def verify_existing() -> None:
            if destination.is_symlink() or not destination.is_file():
                raise FileExistsError(f"Refusing to replace a non-regular prototype artifact: {destination}")
            try:
                existing = type(self).load(destination)
            except (ValueError, KeyError, TypeError) as exc:
                raise FileExistsError(f"Existing prototype artifact is preserved; use a new v2 path: {destination}") from exc
            if existing.fingerprint() != self.fingerprint():
                raise FileExistsError(f"Different prototype bank already exists and is preserved: {destination}")
        if destination.exists() or destination.is_symlink():
            verify_existing()
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(
                {
                    "format": PROTOTYPE_BANK_FORMAT,
                    "fit_contract": PROTOTYPE_FIT_CONTRACT,
                    "descriptor_policy": REGION_DESCRIPTOR_POLICY,
                    "features": torch.from_numpy(self.features.astype(np.float32)),
                    "standardized_centers": torch.from_numpy(
                        self.standardized_centers.astype(np.float32)
                    ),
                    "descriptor_mean": torch.from_numpy(
                        self.descriptor_mean.astype(np.float32)
                    ),
                    "descriptor_std": torch.from_numpy(
                        self.descriptor_std.astype(np.float32)
                    ),
                    "edge_index": torch.from_numpy(self.edge_index.astype(np.int64)),
                    "training_case_ids": list(self.training_case_ids),
                    "fit_descriptors": torch.from_numpy(self.fit_descriptors.astype(np.float32)),
                    "fit_labels": torch.from_numpy(self.fit_labels.astype(np.int64)),
                    "fit_counts": torch.from_numpy(self.fit_counts.astype(np.int64)),
                    "fit_case_counts": torch.from_numpy(self.fit_case_counts.astype(np.int64)),
                    "fit_provenance": copy.deepcopy(self.fit_provenance),
                },
                temporary,
            )
            try:
                os.link(temporary, destination)
            except FileExistsError:
                verify_existing()
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeBank":
        payload = _load_bank_payload(path)
        if not isinstance(payload, dict) or payload.get("format") != PROTOTYPE_BANK_FORMAT:
            raise ValueError("Unsupported or legacy prototype bank format; v1 is read-only audit, not a current bank")
        if payload.get("fit_contract") != PROTOTYPE_FIT_CONTRACT or payload.get("descriptor_policy") != REGION_DESCRIPTOR_POLICY:
            raise ValueError("Prototype bank lacks the current explicit fit/descriptor policy")
        required = {"format", "fit_contract", "descriptor_policy", "features", "standardized_centers",
                    "descriptor_mean", "descriptor_std", "edge_index", "training_case_ids",
                    "fit_descriptors", "fit_labels", "fit_counts", "fit_case_counts", "fit_provenance"}
        if set(payload) != required:
            raise ValueError(f"Invalid v2 prototype payload fields: missing={sorted(required - set(payload))}, extra={sorted(set(payload) - required)}")
        bank = cls(
            features=_payload_array(payload, "features", np.float32),
            standardized_centers=_payload_array(payload, "standardized_centers", np.float32),
            descriptor_mean=_payload_array(payload, "descriptor_mean", np.float32),
            descriptor_std=_payload_array(payload, "descriptor_std", np.float32),
            edge_index=_payload_array(payload, "edge_index", np.int64),
            training_case_ids=tuple(payload["training_case_ids"]),
            fit_descriptors=_payload_array(payload, "fit_descriptors", np.float32),
            fit_labels=_payload_array(payload, "fit_labels", np.int64),
            fit_counts=_payload_array(payload, "fit_counts", np.int64),
            fit_case_counts=_payload_array(payload, "fit_case_counts", np.int64),
            fit_provenance=copy.deepcopy(payload["fit_provenance"]),
        )
        bank.validate(verify_fit=True)
        return bank


def _prototype_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_knn(bank: PrototypeBank) -> int:
    """Check stored standardized-space KNN, allowing tied-neighbor order."""
    src, dst = bank.edge_index
    counts = np.bincount(dst, minlength=bank.num_prototypes)
    if not len(counts) or np.any(counts != counts[0]) or not 1 <= counts[0] < bank.num_prototypes:
        raise ValueError("Prototype KNN must retain a positive fixed degree for every prototype")
    if np.any(src == dst) or len(np.unique(bank.edge_index.T, axis=0)) != len(src):
        raise ValueError("Prototype KNN contains self or duplicate edges")
    distances = np.linalg.norm(bank.standardized_centers[:, None] - bank.standardized_centers[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    for destination in range(bank.num_prototypes):
        chosen = src[dst == destination]
        omitted = np.ones(bank.num_prototypes, dtype=bool)
        omitted[chosen] = False
        if np.any(distances[destination, omitted] < distances[destination, chosen].max()):
            raise ValueError("Stored prototype edges are not standardized-space nearest neighbors")
    return int(counts[0])


def audit_prototype_bank(path: str | Path) -> dict:
    """Read-only numerical sub-proof, never cohort/checkpoint authorization.

    Legacy v1 lacks fit rows, so membership is explicitly unverified. Current
    v2 is checked against every stored fit row. Neither replaces the caller's
    source-image, training-split, model-version, and metadata integrity guards.
    """
    source = Path(path)
    before = _prototype_file_sha256(source)
    payload = _load_bank_payload(source)
    if not isinstance(payload, dict):
        raise ValueError("Invalid prototype bank payload")
    if payload.get("format") == "hiercp_prototype_bank_v1":
        report = audit_legacy_bank(source)
    elif payload.get("format") == PROTOTYPE_BANK_FORMAT:
        bank = PrototypeBank.load(source)
        standardized = (bank.fit_descriptors - bank.descriptor_mean[None]) / bank.descriptor_std[None]
        gaps = [float(np.linalg.norm(standardized[bank.fit_labels == i].mean(axis=0) - bank.standardized_centers[i]))
                for i in range(bank.num_prototypes)]
        report = {
            "format": "hiercp_prototype_bank_audit_v1", "bank_format": PROTOTYPE_BANK_FORMAT,
            "source_sha256": before, "prototype_fingerprint": bank.fingerprint(),
            "prototype_count": bank.num_prototypes, "training_case_ids": list(bank.training_case_ids),
            "fit_descriptor_count": len(bank.fit_descriptors), "knn_degree": _audit_knn(bank),
            "membership_verified": True, "membership_scope": "all_stored_training_descriptors",
            "current_schema_compatible": True, "can_use_for_current_training": False,
            "requires_external_training_cohort_provenance": True, "artifact_modified": False,
            "fit_provenance": copy.deepcopy(bank.fit_provenance),
            "max_final_membership_centroid_gap_l2": max(gaps),
            "not_verifiable_without_fit_descriptors": [],
            "external_checks_required": ["source_case_content_and_order", "declared_training_split", "model_and_cache_contracts"],
        }
    else:
        raise ValueError(f"Unsupported prototype audit format: {payload.get('format')!r}")
    if _prototype_file_sha256(source) != before or report["source_sha256"] != before:
        raise RuntimeError("Prototype artifact changed during read-only audit")
    return report


def audit_legacy_bank(path: str | Path) -> dict:
    """Inspect a v1 artifact without returning a scorer-usable current bank.

    v1 did not store fitting descriptors/labels. Structural checks cannot certify
    final-membership support/dispersion or silently refit/relabel that history.
    """
    source = Path(path)
    before = _prototype_file_sha256(source)
    payload = _load_bank_payload(source)
    if not isinstance(payload, dict) or payload.get("format") != "hiercp_prototype_bank_v1":
        raise ValueError("Legacy audit requires an original hiercp_prototype_bank_v1 payload")
    bank = PrototypeBank(
        features=_payload_array(payload, "features", np.float32),
        standardized_centers=_payload_array(payload, "standardized_centers", np.float32),
        descriptor_mean=_payload_array(payload, "descriptor_mean", np.float32),
        descriptor_std=_payload_array(payload, "descriptor_std", np.float32),
        edge_index=_payload_array(payload, "edge_index", np.int64),
        training_case_ids=tuple(payload["training_case_ids"]),
    )
    bank.validate()
    degree = _audit_knn(bank)
    if not np.allclose(bank.features[:, :REGION_FEATURE_DIM], bank.standardized_centers * bank.descriptor_std[None] + bank.descriptor_mean[None], rtol=1e-6, atol=1e-6):
        raise ValueError("Legacy raw and standardized centers disagree")
    support, dispersion = bank.features[:, REGION_FEATURE_DIM], bank.cluster_mean_distance
    if np.any(support < 0) or not np.isclose(support.sum(), 1, rtol=1e-6, atol=1e-6) or np.any(dispersion < 0):
        raise ValueError("Legacy stored support/dispersion violates numeric invariants")
    if _prototype_file_sha256(source) != before:
        raise RuntimeError("Legacy prototype artifact changed during read-only audit")
    return {
        "format": "hiercp_prototype_bank_audit_v1", "bank_format": payload["format"],
        "original_format": payload["format"], "source_sha256": before,
        "can_use_for_current_training": False, "artifact_modified": False,
        "membership_verified": False, "membership_scope": "unavailable_in_legacy_v1",
        "current_schema_compatible": False, "requires_external_training_cohort_provenance": True,
        "knn_degree": degree,
        "prototype_count": bank.num_prototypes, "training_case_ids": list(bank.training_case_ids),
        "checked": ["shape_finiteness_scaler", "raw_standardized_center_relation", "stored_support_normalization", "stored_dispersion_nonnegative", "standardized_space_knn"],
        "not_verifiable_without_fit_descriptors": ["final_membership_support_and_dispersion", "training_descriptor_provenance", "Lloyd_convergence"],
    }


def _prototype_knn(centers: np.ndarray, k: int) -> np.ndarray:
    count = int(centers.shape[0])
    if isinstance(k, (bool, np.bool_)) or int(k) != k:
        raise ValueError(f"prototype_k must be an exact integer, got {k!r}")
    effective = int(k)
    if count <= 1 or not 1 <= effective < count:
        raise ValueError(
            "prototype_k must satisfy 1 <= prototype_k < num_prototypes; "
            f"prototype_k={effective}, num_prototypes={count}"
        )
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=effective - 1, axis=1)[:, :effective]
    destination = np.repeat(np.arange(count, dtype=np.int64), effective)
    source = neighbors.reshape(-1).astype(np.int64)
    return np.stack([source, destination], axis=0)


def build_prototype_bank(
    case_region_features: Sequence[tuple[str, np.ndarray]],
    *,
    config: GraphBuildConfig,
    rng: np.random.Generator,
) -> PrototypeBank:
    """Fit population region prototypes from training cases only."""

    if not case_region_features:
        raise ValueError("No region feature groups were supplied")
    case_ids = tuple(str(case_id) for case_id, _ in case_region_features)
    if len(set(case_ids)) != len(case_ids) or any(not value or value != value.strip() for value in case_ids):
        raise ValueError("Prototype training case IDs must be nonempty and unique")
    matrices = [np.asarray(values, dtype=np.float32) for _, values in case_region_features]
    if any(matrix.ndim != 2 or matrix.shape[1] != REGION_FEATURE_DIM for matrix in matrices):
        raise ValueError("Every region descriptor matrix must have REGION_FEATURE_DIM columns")
    if any(not len(matrix) or not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("Every training case requires nonempty finite region descriptors")
    all_features = np.concatenate(matrices, axis=0).astype(np.float32)
    mean = all_features.mean(axis=0).astype(np.float32)
    std = all_features.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
    standardized = (all_features - mean[None]) / std[None]
    rng_initial_state = _json_value(copy.deepcopy(rng.bit_generator.state))
    centers, labels, diagnostics = numpy_kmeans(
        standardized,
        config.num_prototypes,
        rng=rng,
        iterations=config.prototype_lloyd_iters,
        return_diagnostics=True,
    )
    cluster_count = int(centers.shape[0])
    raw_centers = centers * std[None] + mean[None]
    support = np.zeros(cluster_count, dtype=np.float32)
    dispersion = np.zeros(cluster_count, dtype=np.float32)
    for cluster in range(cluster_count):
        members = standardized[labels == cluster]
        support[cluster] = float(members.shape[0] / max(1, standardized.shape[0]))
        if members.size:
            dispersion[cluster] = float(
                np.mean(np.linalg.norm(members - centers[cluster][None], axis=1))
            )
    prototype_features = np.concatenate(
        [raw_centers.astype(np.float32), support[:, None], dispersion[:, None]],
        axis=1,
    ).astype(np.float32)
    bank = PrototypeBank(
        features=prototype_features,
        standardized_centers=centers.astype(np.float32),
        descriptor_mean=mean,
        descriptor_std=std,
        edge_index=_prototype_knn(centers, config.prototype_k),
        training_case_ids=case_ids,
        fit_descriptors=all_features.copy(),
        fit_labels=labels.astype(np.int64),
        fit_counts=np.bincount(labels, minlength=cluster_count).astype(np.int64),
        fit_case_counts=np.asarray([len(matrix) for matrix in matrices], dtype=np.int64),
        fit_provenance={
            "fit_contract": PROTOTYPE_FIT_CONTRACT,
            "descriptor_policy": REGION_DESCRIPTOR_POLICY,
            "max_iterations": int(config.prototype_lloyd_iters),
            "assignment_tie_policy": "lowest_prototype_index",
            "clusters": cluster_count,
            "prototype_k": int(config.prototype_k),
            "rng_initial_state": rng_initial_state,
            "numpy_version": np.__version__,
            "input_descriptor_sha256": _array_fingerprint(all_features),
            **diagnostics,
        },
    )
    bank.validate(verify_fit=True)
    return bank
