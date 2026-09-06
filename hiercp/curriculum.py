"""Negative-curriculum candidate selection and relation corruption."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from hiercp.common import (
    CandidateInfo,
    CandidatePreparationError,
    LoadedCase,
    SourceTumor,
    context_stats_for_local_mask,
    distance_to_mask_mm,
)
from hiercp.prototype import PrototypeBank
from hiercp.region import PatientRegionData
from hiercp.schema import (
    CORRUPTION_ANISOTROPIC_SCALE,
    CORRUPTION_NONE,
    CORRUPTION_ORIENTATION,
    CORRUPTION_THICKNESS_SCALE,
    DIFFICULTY_EASY,
    DIFFICULTY_INTER_REGION,
    DIFFICULTY_INTRA_CORRUPTED,
    DIFFICULTY_POSITIVE,
    GraphBuildConfig,
)


@dataclass(frozen=True)
class CandidateSpec:
    center: tuple[int, int, int]
    difficulty: int
    corruption: int
    region_id: int
    prototype_id: int
    liver_coverage: float
    border_distance_mm: float
    occupied_distance_mm: float
    context_mean_hu: float
    context_std_hu: float
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation: tuple[float, ...] = tuple(np.eye(3, dtype=np.float32).reshape(-1).tolist())

    @property
    def scale_array(self) -> np.ndarray:
        return np.asarray(self.scale_xyz, dtype=np.float32)

    @property
    def rotation_matrix(self) -> np.ndarray:
        values = np.asarray(self.rotation, dtype=np.float32)
        if values.size != 9:
            raise ValueError("rotation must contain nine values")
        return values.reshape(3, 3)


def _rotation_matrix(axis: int, angle: float) -> np.ndarray:
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    if axis == 0:
        return np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=np.float32)
    if axis == 1:
        return np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=np.float32)
    return np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float32)


def _source_context(
    case: LoadedCase,
    source: SourceTumor,
    regions: PatientRegionData,
) -> tuple[float, float]:
    organ_patch = regions.full_organ_mask[source.patch_slices]
    return context_stats_for_local_mask(
        source.patch_image,
        organ_patch,
        source.patch_mask,
        ring_width=3,
    )


def _context_distance(
    candidate: CandidateInfo,
    source_mean: float,
    source_std: float,
    source_depth: float,
) -> float:
    return float(
        abs(candidate.context_mean_hu - source_mean) / 50.0
        + abs(candidate.context_std_hu - source_std) / 35.0
        + abs(candidate.border_distance_mm - source_depth) / 30.0
        + (1.0 - candidate.liver_coverage) * 2.0
    )


def _base_spec(
    candidate: CandidateInfo,
    *,
    difficulty: int,
    region_id: int,
    prototype_id: int,
) -> CandidateSpec:
    return CandidateSpec(
        center=tuple(int(value) for value in candidate.center),
        difficulty=int(difficulty),
        corruption=CORRUPTION_NONE,
        region_id=int(region_id),
        prototype_id=int(prototype_id),
        liver_coverage=float(candidate.liver_coverage),
        border_distance_mm=float(candidate.border_distance_mm),
        occupied_distance_mm=float(candidate.occupied_distance_mm),
        context_mean_hu=float(candidate.context_mean_hu),
        context_std_hu=float(candidate.context_std_hu),
    )


def _corrupt(spec: CandidateSpec, index: int, rng: np.random.Generator) -> CandidateSpec:
    """Break exactly one tumor-context relation for an intra-region negative."""

    mode = index % 3
    if mode == 0:
        scale = float(rng.uniform(1.30, 1.55))
        return replace(
            spec,
            corruption=CORRUPTION_THICKNESS_SCALE,
            scale_xyz=(scale, scale, scale),
        )
    if mode == 1:
        values = np.asarray(
            [rng.uniform(1.35, 1.60), rng.uniform(0.65, 0.82), rng.uniform(0.95, 1.10)],
            dtype=np.float32,
        )
        rng.shuffle(values)
        return replace(
            spec,
            corruption=CORRUPTION_ANISOTROPIC_SCALE,
            scale_xyz=tuple(float(value) for value in values),
        )
    rotation = _rotation_matrix(
        int(rng.integers(0, 3)),
        float(rng.choice([np.pi / 2, -np.pi / 2])),
    )
    return replace(
        spec,
        corruption=CORRUPTION_ORIENTATION,
        rotation=tuple(float(value) for value in rotation.reshape(-1)),
    )


def _category_counts(
    negative_count: int,
    easy_fraction: float,
    inter_fraction: float,
    intra_fraction: float,
) -> tuple[int, int, int]:
    fractions = np.asarray([easy_fraction, inter_fraction, intra_fraction], dtype=np.float64)
    if np.any(fractions < 0) or float(fractions.sum()) <= 0:
        raise ValueError("Curriculum fractions must be non-negative and non-zero")
    fractions /= fractions.sum()
    counts = np.floor(fractions * negative_count).astype(int)
    while int(counts.sum()) < negative_count:
        residual = fractions * negative_count - counts
        counts[int(np.argmax(residual))] += 1
    return tuple(int(value) for value in counts)


def build_training_specs(
    case: LoadedCase,
    source: SourceTumor,
    candidates: Sequence[CandidateInfo],
    regions: PatientRegionData,
    bank: PrototypeBank,
    *,
    total_candidates: int,
    easy_fraction: float,
    inter_fraction: float,
    intra_fraction: float,
    tumor_label: int,
    config: GraphBuildConfig,
    rng: np.random.Generator,
) -> list[CandidateSpec]:
    """Build positive + easy + inter-region + relation-corrupted negatives.

    A candidate in the source region or source prototype is never used as a
    negative without an explicit geometry/context corruption.
    """

    if total_candidates < 4:
        raise ValueError("total_candidates must be at least four")
    if len(candidates) < total_candidates - 1:
        raise CandidatePreparationError(
            "insufficient_valid_candidate_pool",
            {"available_candidates": len(candidates),
             "required_negative_candidates": int(total_candidates) - 1},
            message="The complete ranking sample cannot be formed from the available pool")

    source_region = regions.region_at(source.anchor_center)
    assignments, _ = bank.assign(
        regions.region_features,
        top_k=config.prototype_top_m,
        temperature=config.prototype_temperature,
    )
    primary_prototype = assignments[:, 0]
    source_prototype = int(primary_prototype[source_region])
    source_mean, source_std = _source_context(case, source, regions)
    source_depth = float(regions.organ_depth[source.anchor_center])
    occupied_without_source = (case.label == int(tumor_label)) & ~source.full_mask
    distance_to_other = distance_to_mask_mm(occupied_without_source, case.spacing)

    positive = CandidateSpec(
        center=source.anchor_center,
        difficulty=DIFFICULTY_POSITIVE,
        corruption=CORRUPTION_NONE,
        region_id=source_region,
        prototype_id=source_prototype,
        liver_coverage=1.0,
        border_distance_mm=source_depth,
        occupied_distance_mm=float(distance_to_other[source.anchor_center]),
        context_mean_hu=source_mean,
        context_std_hu=source_std,
    )

    records: list[dict[str, object]] = []
    for candidate in candidates:
        region_id = regions.region_at(candidate.center)
        records.append(
            {
                "candidate": candidate,
                "region": region_id,
                "prototype": int(primary_prototype[region_id]),
                "distance": _context_distance(candidate, source_mean, source_std, source_depth),
            }
        )

    negative_count = total_candidates - 1
    easy_target, inter_target, intra_target = _category_counts(
        negative_count,
        easy_fraction,
        inter_fraction,
        intra_fraction,
    )
    used: set[tuple[int, int, int]] = set()

    def take(
        pool: Sequence[dict[str, object]],
        count: int,
        *,
        reverse: bool,
    ) -> list[dict[str, object]]:
        ordered = sorted(pool, key=lambda item: float(item["distance"]), reverse=reverse)
        output: list[dict[str, object]] = []
        for item in ordered:
            candidate = item["candidate"]
            assert isinstance(candidate, CandidateInfo)
            center = tuple(candidate.center)
            if center in used:
                continue
            used.add(center)
            output.append(item)
            if len(output) >= count:
                break
        return output

    easy = take(
        [record for record in records if int(record["prototype"]) != source_prototype],
        easy_target,
        reverse=True,
    )
    inter = take(
        [
            record
            for record in records
            if int(record["region"]) != source_region
            and int(record["prototype"]) != source_prototype
        ],
        inter_target,
        reverse=False,
    )
    intra = take(
        [record for record in records if int(record["region"]) == source_region],
        intra_target,
        reverse=False,
    )
    if len(intra) < intra_target:
        intra.extend(
            take(
                [
                    record
                    for record in records
                    if int(record["prototype"]) == source_prototype
                ],
                intra_target - len(intra),
                reverse=False,
            )
        )

    labeled: list[tuple[int, dict[str, object], bool]] = []
    labeled.extend((DIFFICULTY_EASY, record, False) for record in easy)
    labeled.extend((DIFFICULTY_INTER_REGION, record, False) for record in inter)
    labeled.extend((DIFFICULTY_INTRA_CORRUPTED, record, True) for record in intra)

    # Fill category shortages safely.  Same-region/same-prototype items are
    # always relation-corrupted; never silently converted into plain negatives.
    remaining = [
        record
        for record in sorted(records, key=lambda item: float(item["distance"]))
        if isinstance(record["candidate"], CandidateInfo)
        and tuple(record["candidate"].center) not in used
    ]
    for record in remaining:
        if len(labeled) >= negative_count:
            break
        candidate = record["candidate"]
        assert isinstance(candidate, CandidateInfo)
        used.add(tuple(candidate.center))
        relation_close = (
            int(record["region"]) == source_region
            or int(record["prototype"]) == source_prototype
        )
        if relation_close:
            labeled.append((DIFFICULTY_INTRA_CORRUPTED, record, True))
        else:
            labeled.append((DIFFICULTY_INTER_REGION, record, False))

    if len(labeled) < negative_count:
        raise CandidatePreparationError(
            "insufficient_distinct_curriculum_candidates",
            {"available_candidates": len(candidates), "distinct_centers": len(used),
             "required_negative_candidates": int(negative_count),
             "formed_negative_candidates": len(labeled),
             "category_targets": {"easy": int(easy_target), "inter": int(inter_target),
                                  "intra_corrupted": int(intra_target)},
             "formed_categories": {"easy": len(easy), "inter": len(inter),
                                   "intra_corrupted": len(intra)}},
            message="Distinct candidate centers are insufficient after curriculum assignment")

    specs: list[CandidateSpec] = [positive]
    corruption_index = 0
    for difficulty, record, requires_corruption in labeled[:negative_count]:
        candidate = record["candidate"]
        assert isinstance(candidate, CandidateInfo)
        base = _base_spec(
            candidate,
            difficulty=difficulty,
            region_id=int(record["region"]),
            prototype_id=int(record["prototype"]),
        )
        if requires_corruption:
            base = _corrupt(base, corruption_index, rng)
            corruption_index += 1
        specs.append(base)
    return specs if len(specs) == total_candidates else None



def build_generation_specs(
    candidates: Sequence[CandidateInfo],
    regions: PatientRegionData,
    bank: PrototypeBank,
    *,
    config: GraphBuildConfig,
) -> list[CandidateSpec]:
    assignments, _ = bank.assign(
        regions.region_features,
        top_k=config.prototype_top_m,
        temperature=config.prototype_temperature,
    )
    primary = assignments[:, 0]
    output: list[CandidateSpec] = []
    for candidate in candidates:
        region_id = regions.region_at(candidate.center)
        output.append(
            _base_spec(
                candidate,
                difficulty=DIFFICULTY_EASY,
                region_id=region_id,
                prototype_id=int(primary[region_id]),
            )
        )
    return output
