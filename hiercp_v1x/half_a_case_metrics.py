"""Optional raw per-case validation score observation for preserved v1 half A.

The recorder observes the existing eval-mode forward exactly once. It performs
one detached CPU transfer of the stacked [samples,8] scores per validation batch,
then writes the actual case IDs and unmodified scores. It does not run a model,
compute ranks, change tie handling, replace the original sample-weighted metrics,
or classify model quality. Every invocation owns a new exclusive JSONL file;
an interrupted partial pass remains visibly incomplete in ``totals()``.
"""
from __future__ import annotations

from collections import Counter
import functools
import json
from pathlib import Path
import re
import uuid

import torch


FORMAT = "hiercp_half_a_validation_case_scores_v1"
_INSTALL_ATTRIBUTE = "_half_a_case_score_recorder"


class CaseScoreRecorder:
    """Record original GT index0 and exactly eight actual candidate scores."""

    def __init__(self, output_dir, session, *, expected_case_ids,
                 expected_samples: int, invocation_id: str | None = None):
        if (not isinstance(expected_case_ids, (list, tuple)) or not expected_case_ids
                or any(not isinstance(case, str) or not case for case in expected_case_ids)
                or len(set(expected_case_ids)) != len(expected_case_ids)):
            raise ValueError("Unique actual validation case IDs are required")
        if type(expected_samples) is not int or expected_samples < len(expected_case_ids):
            raise ValueError("Expected validation sample count must cover every configured case")
        if not hasattr(session, "_epoch"):
            raise ValueError("An actual EpochTelemetry session is required")
        invocation = uuid.uuid4().hex if invocation_id is None else invocation_id
        if not isinstance(invocation, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", invocation) is None:
            raise ValueError("Validation score invocation ID must be a safe unique filename component")
        self.expected_case_ids = tuple(expected_case_ids)
        self.expected_samples = expected_samples
        self.invocation_id = invocation
        self.session = session
        self._epochs: dict[int, Counter] = {}
        self._last_epoch: int | None = None
        self.path = Path(output_dir).resolve() / f"validation_case_scores_{invocation}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Existing logs/results are never replaced or appended across invocations.
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")

    def _epoch(self) -> int:
        current = self.session._epoch
        if current is None:
            return 0
        if type(current) is not int or current < 1:
            raise ValueError("EpochTelemetry validation epoch must be None or a positive integer")
        return current

    def record(self, batch, output) -> None:
        if self._stream.closed:
            raise ValueError("Validation score recorder was closed before eval forward")
        epoch = self._epoch()
        if self._last_epoch is not None and epoch != self._last_epoch:
            previous = self._epochs[self._last_epoch]
            if sum(previous.values()) != self.expected_samples:
                raise ValueError("Validation epoch changed before all expected samples were recorded")
            if epoch <= self._last_epoch:
                raise ValueError("Validation epoch moved backward or repeated initial validation")
        case_ids = getattr(batch, "case_ids", None)
        counts = getattr(batch, "counts", None)
        scores = getattr(output, "scores", None)
        if (not isinstance(case_ids, (list, tuple)) or not case_ids
                or not isinstance(counts, (list, tuple)) or len(counts) != len(case_ids)
                or any(type(count) is not int or count != 8 for count in counts)
                or not isinstance(scores, (list, tuple)) or len(scores) != len(case_ids)):
            raise ValueError("Actual batch case IDs/counts and one eight-score vector per sample are required")
        expected = set(self.expected_case_ids)
        if any(not isinstance(case, str) or case not in expected for case in case_ids):
            raise ValueError("Eval batch contains a case outside the complete validation split")
        reference = scores[0]
        if not isinstance(reference, torch.Tensor):
            raise ValueError("Actual floating validation tensors are required")
        for value in scores:
            if (not isinstance(value, torch.Tensor) or value.shape != (8,)
                    or not value.is_floating_point() or value.device != reference.device
                    or value.dtype != reference.dtype):
                raise ValueError("Eight floating scores on one device/dtype required for every actual sample")
        counter = self._epochs.get(epoch, Counter())
        next_counter = counter + Counter(case_ids)
        sample_count = sum(next_counter.values())
        if sample_count > self.expected_samples:
            raise ValueError("Validation score recorder received more than the configured complete sample count")
        if sample_count == self.expected_samples and set(next_counter) != expected:
            raise ValueError("Complete validation sample count does not cover every configured validation case")
        # The only score CPU copy/synchronization. No gradient, parameter, input,
        # optimizer, or original metric state is changed by this observer.
        score_cpu = torch.stack(tuple(value.detach() for value in scores), dim=0).cpu()
        if not bool(torch.isfinite(score_cpu).all()):
            raise ValueError("Nonfinite actual validation scores cannot be recorded")
        vectors = score_cpu.tolist()
        rows = []
        written = counter.copy()
        for case, vector in zip(case_ids, vectors):
            written[case] += 1
            rows.append({
                "format": FORMAT, "invocation_id": self.invocation_id,
                "epoch": epoch, "case_id": case,
                "sample_ordinal_in_case_epoch": written[case],
                "gt_index": 0, "candidate_count": 8, "scores": vector,
                "expected_validation_samples": self.expected_samples,
                "expected_validation_cases": len(self.expected_case_ids),
                "sample_weighted_metrics_changed": False,
            })
        # Validate the complete batch's JSON before touching its output file.
        lines = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
        self._stream.write(lines)
        self._stream.flush()
        self._epochs[epoch] = next_counter
        self._last_epoch = epoch

    def totals(self) -> dict:
        """Coverage metadata only, never accuracy or a quality classification."""
        return {
            "format": FORMAT, "invocation_id": self.invocation_id, "path": str(self.path),
            "expected_case_ids": list(self.expected_case_ids), "expected_samples": self.expected_samples,
            "epochs": {str(epoch): {
                "samples": sum(counter.values()),
                "cases": [case for case in self.expected_case_ids if counter[case]],
                "case_sample_counts": {case: counter[case] for case in self.expected_case_ids},
                "complete": (sum(counter.values()) == self.expected_samples
                             and set(counter) == set(self.expected_case_ids)),
            } for epoch, counter in sorted(self._epochs.items())},
        }

    def close(self) -> None:
        self._stream.close()


def install_case_score_recorder(model_class, session, output_dir, *, expected_case_ids,
                               expected_samples: int) -> CaseScoreRecorder:
    """Runtime-only optional observer; restore()/close() belong to the caller.

    Install after creating EpochTelemetry and before the original training loop.
    Original ``forward`` receives exactly its existing arguments and its output
    object is returned unchanged. Training forwards incur no score recording.
    """
    if not isinstance(model_class, type) or not callable(getattr(model_class, "forward", None)):
        raise ValueError("Original model class with a callable forward is required")
    if hasattr(model_class, _INSTALL_ATTRIBUTE):
        raise ValueError("Validation case score observer is already installed on this model class")
    recorder = CaseScoreRecorder(output_dir, session, expected_case_ids=expected_case_ids,
                                 expected_samples=expected_samples)
    original = model_class.forward

    @functools.wraps(original)
    def observed(self, batch, *args, **kwargs):
        output = original(self, batch, *args, **kwargs)
        if not self.training:
            recorder.record(batch, output)
        return output

    model_class.forward = observed
    setattr(model_class, _INSTALL_ATTRIBUTE, recorder)

    def restore() -> None:
        if model_class.forward is not observed:
            raise ValueError("Model forward changed after installing the validation recorder")
        model_class.forward = original
        delattr(model_class, _INSTALL_ATTRIBUTE)
    recorder.restore = restore
    return recorder
