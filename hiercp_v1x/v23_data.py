"""v2.3 complete observed-P, frozen-donor, cumulative native-U contracts.

This module selects observations, never constructs smaller graphs or redraws
donors. Original CT inputs and full graph construction belong to the existing
native providers. Targets are retained for the loss, removed from upper queries.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

from .transition_evaluation import QUERY_FIELDS, validate_cohort


FORMAT = "v23_all_observed_P_cumulative_native_U_v1"
ZERO_P_TRAIN_POLICY = "retained_in_inventory_no_defined_P_U_ranking_loss"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def cumulative_u_counts(initial=7, expansion=7):
    """The declared stage ladder; reaching the end never discards earlier U."""
    if (type(initial) is not int or type(expansion) is not int
            or not 1 <= initial <= 128 or not 1 <= expansion <= 128):
        raise ValueError("Explicit positive native-U stage sizes within full128 required")
    values = list(range(initial, 129, expansion))
    if values[-1] != 128:
        values.append(128)
    return tuple(values)


@dataclass(frozen=True)
class CasePlan:
    """One complete active case; positions refer to its actual joint forward."""
    case_id: str
    partition: str
    patient_group: str
    active_u_count: int
    record_ids: tuple[str, ...]
    inventory_indices: tuple[int, ...]
    positive_indices: tuple[int, ...]
    unobserved_indices: tuple[int, ...]
    unobserved_bank_positions: tuple[int, ...]
    bank_record_ids: tuple[str, ...]
    donor_case_id: str
    donor_component: int
    donor_group: str
    _rows: tuple[dict, ...] = field(repr=False, compare=False)

    @property
    def rows(self):
        """Defensive copies: consumers cannot mutate the signed population."""
        return copy.deepcopy(list(self._rows))

    @property
    def query_rows(self):
        """The exact existing target-free query schema and candidate order."""
        return [{key: copy.deepcopy(row[key]) for key in QUERY_FIELDS} for row in self._rows]

    @property
    def observed_P(self):
        return len(self.positive_indices)

    @property
    def indices(self):
        return self.inventory_indices

    def manifest(self):
        value = dict(case_id=self.case_id, partition=self.partition, patient_group=self.patient_group,
                     active_U=self.active_u_count, observed_P=self.observed_P,
                     record_ids=list(self.record_ids),
                     positive_indices=list(self.positive_indices),
                     unobserved_indices=list(self.unobserved_indices),
                     unobserved_bank_positions=list(self.unobserved_bank_positions),
                     fixed_donor=dict(case_id=self.donor_case_id,
                                      component=self.donor_component, group=self.donor_group),
                     query_rows_sha256=_digest(self.query_rows),
                     all_P_retained=True, P_as_negative=False, donor_redraw=False)
        value["sha256"] = _digest(value)
        return value


class V23Population:
    """Signed complete native population reused by train and both validations.

    U bank position is defined by raw ``comparison.centers``, as it was in the
    comparison experiments. Lexicographically sorted observation IDs are not a
    replacement bank order. Selected rows retain original inventory order so
    full128 queries remain identical to the sealed native hard evaluation.
    """
    def __init__(self, inventory, *, debug=False):
        if not isinstance(inventory, Mapping) or type(debug) is not bool:
            raise ValueError("Explicit native inventory and boolean DEBUG required")
        meta = copy.deepcopy(dict(inventory))
        cohort = validate_cohort(meta, debug=debug)
        self._meta = meta
        self.debug = debug
        self._records = tuple(copy.deepcopy(meta["records"]))
        self._case_ids = {partition: tuple(meta["split"][partition])
                          for partition in ("inner_train", "inner_val", "outer_train")}
        self._by_case = {case: tuple(i for i, row in enumerate(self._records)
                                    if row["case_id"] == case)
                         for case in self._case_ids["outer_train"]}
        raw = {row["case_id"]: row for row in meta["raw_records"]}
        self._bank_ids = {}
        self._positive_counts = {}
        self._case_groups = {}
        for case, indices in self._by_case.items():
            rows = [self._records[i] for i in indices]
            positives = [row for row in rows if row["target"] == 1]
            if len({row["component"] for row in positives}) != len(positives):
                raise ValueError("Each actual observed P component must appear exactly once: " + case)
            by_center = {tuple(row["center"]): row["id"] for row in rows if row["target"] == 0}
            bank = tuple(by_center[tuple(center)] for center in raw[case]["comparison"]["centers"])
            if len(bank) != 128 or len(set(bank)) != 128:
                raise ValueError("Frozen complete native128 bank required: " + case)
            self._bank_ids[case] = bank
            self._positive_counts[case] = len(positives)
            self._case_groups[case] = rows[0]["patient_group"]
        counts = {partition: self._counts(self._case_ids[partition])
                  for partition in ("inner_train", "inner_val")}
        if not debug and (
                (counts["inner_train"]["cases"], counts["inner_train"]["observed_P"],
                 counts["inner_train"]["unobserved_U"]) != (84, 527, 10752)
                or (counts["inner_val"]["cases"], counts["inner_val"]["observed_P"],
                    counts["inner_val"]["unobserved_U"]) != (21, 135, 2688)
                or len(self._records) != 14102):
            raise ValueError("v2.3 production requires full84/P527/U10752 train and full21/P135/U2688 validation")
        self._manifest = dict(
            format=FORMAT, version="v2.3", DEBUG=debug, seed=42,
            inventory_metadata_sha256=cohort["inventory_metadata_sha256"],
            native_assignment=copy.deepcopy(cohort["native_assignment"]),
            all_observed_P_active_from_start=True, P_as_negative=False,
            U_semantics="unobserved comparison locations; not CP-unsuitable GT",
            U_bank_size=128, U_bank_order="raw_records.comparison.centers unchanged",
            query_order="original signed inventory rows unchanged after subset selection",
            U_order_is_measured_difficulty=False, cumulative_U_replay=True,
            frozen_bank_sha256=_digest({case: list(bank) for case, bank in self._bank_ids.items()}),
            partition_cases={key: list(value) for key, value in self._case_ids.items()},
            counts=counts, zero_P_train_policy=ZERO_P_TRAIN_POLICY,
            zero_P_train_cases=[case for case in self._case_ids["inner_train"]
                                if self._positive_counts[case] == 0],
            zero_P_validation_cases=[case for case in self._case_ids["inner_val"]
                                     if self._positive_counts[case] == 0],
            validation_zero_P_cases_scored=True,
            patient_balance="mean_U_per_P then mean_P_per_patient then mean_patients",
            independent_train_only_donor_fixed_per_case=True,
            donor_redraw=False, hidden_subset=False,
            retained_observations=len(self._records), data_fraction=1.)
        self._manifest["sha256"] = _digest(self._manifest)

    def _counts(self, cases):
        p = sum(self._positive_counts[case] for case in cases)
        return dict(cases=len(cases), patient_groups=len({self._case_groups[case] for case in cases}),
                    ranking_cases=sum(self._positive_counts[case] > 0 for case in cases),
                    zero_P_cases=sum(self._positive_counts[case] == 0 for case in cases),
                    observed_P=p, unobserved_U=128 * len(cases), records=p + 128 * len(cases))

    @property
    def meta(self):
        """Original full inventory for the verified native geometry providers."""
        return copy.deepcopy(self._meta)

    @property
    def rows(self):
        return copy.deepcopy(list(self._records))

    @property
    def by_case(self):
        return MappingProxyType(self._by_case)

    @property
    def case_groups(self):
        return MappingProxyType(self._case_groups)

    def manifest(self):
        return copy.deepcopy(self._manifest)

    def partition_cases(self, partition, *, ranking_only=False):
        if partition not in self._case_ids or type(ranking_only) is not bool:
            raise ValueError("Explicit retained native partition and boolean ranking policy required")
        if ranking_only and partition != "inner_train":
            raise ValueError("Zero-P exclusion is an explicit training-loss policy; validation retains every case")
        cases = self._case_ids[partition]
        if ranking_only:
            cases = tuple(case for case in cases if self._positive_counts[case] > 0)
        return cases

    def case(self, case_id, active_u_count=128):
        if case_id not in self._by_case:
            raise ValueError("Case outside signed retained native population: " + str(case_id))
        if type(active_u_count) is not int or not 1 <= active_u_count <= 128:
            raise ValueError("Explicit active native-U count within full128 required")
        bank = self._bank_ids[case_id]
        active = set(bank[:active_u_count])
        indices = tuple(i for i in self._by_case[case_id]
                        if self._records[i]["target"] == 1 or self._records[i]["id"] in active)
        rows = tuple(self._records[i] for i in indices)
        p = tuple(i for i, row in enumerate(rows) if row["target"] == 1)
        u = tuple(i for i, row in enumerate(rows) if row["target"] == 0)
        if len(p) != self._positive_counts[case_id] or len(u) != active_u_count:
            raise ValueError("Cumulative case plan lost an observed P or active U")
        positions = {identity: i for i, identity in enumerate(bank)}
        first = rows[0]
        partition = "inner_train" if case_id in self._case_ids["inner_train"] else "inner_val"
        return CasePlan(case_id=case_id, partition=partition, patient_group=self._case_groups[case_id],
                        active_u_count=active_u_count, record_ids=tuple(row["id"] for row in rows),
                        inventory_indices=indices, positive_indices=p, unobserved_indices=u,
                        unobserved_bank_positions=tuple(positions[rows[i]["id"]] for i in u),
                        bank_record_ids=bank, donor_case_id=first["donor_case_id"],
                        donor_component=first["donor_component"], donor_group=first["donor_group"],
                        _rows=copy.deepcopy(rows))

    def indices_for(self, plan, dataset_or_rows):
        """Map the exact case query order into an original provider partition."""
        if not isinstance(plan, CasePlan):
            raise ValueError("Case plan belongs to another native population")
        expected_plan = self.case(plan.case_id, plan.active_u_count)
        if plan != expected_plan or plan._rows != expected_plan._rows:
            raise ValueError("Case plan belongs to another native population")
        rows = getattr(dataset_or_rows, "rows", dataset_or_rows)
        if not isinstance(rows, (tuple, list)):
            raise ValueError("Original native Dataset or its complete rows required")
        by_id = {row["id"]: (index, row) for index, row in enumerate(rows)}
        if len(by_id) != len(rows) or not set(plan.record_ids) <= set(by_id):
            raise ValueError("Provider rows lack exact distinct active native identities")
        indices = []
        for expected in plan._rows:
            index, actual = by_id[expected["id"]]
            if any(actual.get(key) != value for key, value in expected.items()):
                raise ValueError("Original provider observation or donor changed: " + expected["id"])
            indices.append(index)
        return tuple(indices)

    def stage_manifest(self, active_u_count):
        plans = [self.case(case, active_u_count).manifest()
                 for case in self._case_ids["outer_train"]]
        value = dict(format=FORMAT, population_sha256=self._manifest["sha256"],
                     active_U=active_u_count, cases=plans, all_P_retained=True,
                     donor_redraw=False, joint_upper_subset_forward_required=True,
                     full_score_slicing_for_stage_validation=False)
        value["sha256"] = _digest(value)
        return value
