"""Cross-patient population prototype bank for liver-region descriptors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
import torch

from hiercp.region import numpy_kmeans
from hiercp.tensor import torch_load_compat
from hiercp.schema import PROTOTYPE_FEATURE_DIM, REGION_FEATURE_DIM, GraphBuildConfig


@dataclass
class PrototypeBank:
    """Population prototypes fitted only on the training-case region graphs."""

    features: np.ndarray
    standardized_centers: np.ndarray
    descriptor_mean: np.ndarray
    descriptor_std: np.ndarray
    edge_index: np.ndarray
    training_case_ids: tuple[str, ...]

    @property
    def num_prototypes(self) -> int:
        return int(self.features.shape[0])

    def fingerprint(self) -> str:
        """Content hash used to bind caches/checkpoints to this exact bank."""

        digest = hashlib.sha256()
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
        return digest.hexdigest()

    def validate(self) -> None:
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

    def assign(
        self,
        region_features: np.ndarray,
        *,
        top_k: int,
        temperature: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.validate()
        values = np.asarray(region_features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != REGION_FEATURE_DIM:
            raise ValueError(f"Unexpected region feature matrix: {values.shape}")
        standardized = (values - self.descriptor_mean[None]) / self.descriptor_std[None]
        squared = np.sum(
            (standardized[:, None] - self.standardized_centers[None]) ** 2,
            axis=-1,
        )
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
        indices = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(squared, indices, axis=1)
        order = np.argsort(ordered, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        ordered = np.take_along_axis(ordered, order, axis=1)
        logits = -ordered / temperature_value
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        normalizer = weights.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
            raise FloatingPointError("Prototype assignment produced invalid weights")
        weights /= normalizer
        return indices.astype(np.int64), weights.astype(np.float32)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        self.validate()
        destination = Path(path)
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
                    "format": "hiercp_prototype_bank_v1",
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
                },
                temporary,
            )
            if overwrite:
                os.replace(temporary, destination)
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError as exc:
                    raise FileExistsError(
                        "Refusing to replace an existing prototype bank without "
                        f"explicit overwrite authorization: {destination}"
                    ) from exc
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeBank":
        payload = torch_load_compat(Path(path), map_location="cpu")
        if payload.get("format") != "hiercp_prototype_bank_v1":
            raise ValueError(f"Unsupported prototype bank format: {payload.get('format')}")
        bank = cls(
            features=payload["features"].numpy().astype(np.float32),
            standardized_centers=payload["standardized_centers"].numpy().astype(np.float32),
            descriptor_mean=payload["descriptor_mean"].numpy().astype(np.float32),
            descriptor_std=payload["descriptor_std"].numpy().astype(np.float32),
            edge_index=payload["edge_index"].numpy().astype(np.int64),
            training_case_ids=tuple(str(value) for value in payload["training_case_ids"]),
        )
        bank.validate()
        return bank


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
    matrices = [np.asarray(values, dtype=np.float32) for _, values in case_region_features]
    if any(matrix.ndim != 2 or matrix.shape[1] != REGION_FEATURE_DIM for matrix in matrices):
        raise ValueError("Every region descriptor matrix must have REGION_FEATURE_DIM columns")
    all_features = np.concatenate(matrices, axis=0).astype(np.float32)
    mean = all_features.mean(axis=0).astype(np.float32)
    std = all_features.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
    standardized = (all_features - mean[None]) / std[None]
    centers, labels = numpy_kmeans(
        standardized,
        config.num_prototypes,
        rng=rng,
        iterations=config.prototype_lloyd_iters,
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
    )
    bank.validate()
    return bank
