"""Execution-only GPU policy admission for complete comparison training batches.

Graph inventories are a scheduling heuristic, never an activation-memory proof.
Unknown larger batches use the original execution settings until a measured
high-water probe is warranted. Every original graph and source remains present.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from . import comparison_gpu_policy as policy
from .comparison_gpu_cache import ExecutionPolicyCache, PolicyCacheError, bind_context, measured_entry
from .comparison_storage import report_secondary_failure, storage_error


FORMAT = "comparison_gpu_runtime_v1"
GRAPH_NAMES = ("local_batch", "local_batch_view2", "patient_batch", "prototype_batch")


def workload_inventory(workload):
    """Read the already-measured CPU inventory; never inspect CUDA graph data."""
    if not isinstance(workload, dict):
        raise TypeError("Actual complete training workload inventory required")
    result = {}

    def count(name, value):
        if type(value) is not int or value < 0:
            raise ValueError(f"Nonnegative actual workload integer required: {name}")
        result[name] = value

    for name in ("source_patch_shape", "target_patch_shape"):
        shape = workload.get(name)
        if not isinstance(shape, (list, tuple)) or not shape:
            raise ValueError(f"Complete actual dense input shape required: {name}")
        if any(type(value) is not int or value < 1 for value in shape):
            raise ValueError(f"Positive actual dense input dimensions required: {name}")
        count(name + ".elements", math.prod(shape))
        for index, value in enumerate(shape):
            count(f"{name}.{index}", value)
    counts = workload.get("counts")
    if (not isinstance(counts, (list, tuple)) or not counts
            or any(type(value) is not int or value != 8 for value in counts)):
        raise ValueError("Complete original eight-candidate training workload required")
    count("source_problems", len(counts))
    for name in GRAPH_NAMES:
        graph = workload.get(name)
        if not isinstance(graph, dict):
            raise ValueError(f"Missing complete actual graph inventory: {name}")
        count(name + ".graphs", graph["graphs"])
        for kind in ("nodes", "edges"):
            for metric in ("total", "max"):
                count(f"{name}.{kind}.{metric}", graph[kind][metric])
        for kind in ("node_types", "edge_types"):
            values = graph.get(kind)
            if not isinstance(values, dict) or not values:
                raise ValueError(f"Complete original graph type inventory required: {name}.{kind}")
            for key, value in sorted(values.items()):
                count(f"{name}.{kind}.{key}", value)
    return result


def _signature(inventory):
    return hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ComparisonGpuRuntime:
    """Calibrate execution attributes without changing checkpoint/training state.

    ``ensure`` receives the same complete GPU batch that the next original
    update will consume. The caller owns failed-forward RNG/gradient cleanup;
    this controller never calls optimizer.step or advances a source cursor.
    """

    def __init__(self, net, loss_fn, *, amp, cuda_limit_bytes, output, notify=None,
                 reserve_bytes=None, cache_context=None):
        if not callable(loss_fn):
            raise TypeError("The current arm's original bound loss function is required")
        if type(amp) is not bool or type(cuda_limit_bytes) is not int or cuda_limit_bytes <= 0:
            raise ValueError("Explicit AMP setting and existing CUDA byte budget required")
        self.net, self.loss_fn = net, loss_fn
        self.amp, self.cuda_limit_bytes = amp, cuda_limit_bytes
        self.reserve_bytes, self.notify = reserve_bytes, notify
        self.path = Path(output) / "gpu_execution.jsonl"
        self.original = copy.deepcopy(policy.execution_settings(net))
        self._measured = []
        self._baseline_only = set()
        self._uncached_original_only = set()
        self._unknown_reported = set()
        self._unknown_notified = False
        self._current = dict(reason="original_before_measurement", settings=copy.deepcopy(self.original))
        self._cache = None
        self._rejected = []
        self._loaded_cache_pending = False
        self._event_started = False
        self._audit_failed = False
        if cache_context is not None:
            binding = bind_context(cache_context, amp=amp, cuda_limit_bytes=cuda_limit_bytes,
                                   reserve_bytes=reserve_bytes, original=self.original)
            self._cache = ExecutionPolicyCache(output, binding)
            self._measured, self._rejected = self._cache.load()
            self._baseline_only = {row['workload_sha256'] for row in self._rejected}
            self._loaded_cache_pending = bool(self._measured or self._rejected)
            # Each invocation has a new audit destination. Never append to or
            # infer policies from possibly quota-truncated historical JSONL.
            self.path = Path(output) / 'gpu_execution_events' / (uuid.uuid4().hex + '.jsonl')

    def _record(self, event, **values):
        if self._audit_failed:
            raise RuntimeError(f'GPU audit previously failed; refusing to append to a possibly partial event: {self.path}')
        row = dict(format=FORMAT, event=event, time=time.time(),
                   workload_admission="componentwise inventory heuristic; not worst-case memory proof",
                   model_graphs_source_batch_objective_and_training_state_unchanged=True, **values)
        if self._cache is not None:
            row['cache_context_sha256'] = self._cache.context_sha256
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            mode = 'x' if self._cache is not None and not self._event_started else 'a'
            with self.path.open(mode, encoding="utf8") as stream:
                self._event_started = True
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            if self._cache is not None:
                self._audit_failed = True
            raise storage_error(error, operation=f'GPU execution audit ({event})', path=self.path) from error
        return row

    def _notice(self, event, **values):
        if event == "calibration_started":
            reason = "larger measured workload" if values["new_high_water"] else "first actual batch"
            text = (f"GPU execution calibration | {reason} | full batch {values['source_problems']} sources "
                    "| measuring before optimizer update")
        elif event == "calibration_completed":
            selected = values["settings"]
            text = (f"GPU execution selected | dense chunk={selected['dense_batch_size']} "
                    f"CNN checkpoint={selected['checkpoint_dense_encoder']} "
                    f"L0 checkpoint={selected['checkpoint_local_blocks']}")
        elif event == "unmeasured_larger_workload_original_execution":
            text = ("GPU execution | larger unmeasured workloads use original settings; "
                    "new high-water measurements and full details are recorded in gpu_execution.jsonl")
        elif event == "optimized_execution_OOM_original_retry":
            text = "GPU execution OOM | restoring original execution for the same complete batch"
        else:
            raise ValueError("Unknown GPU execution notice")
        if self.notify is not None:
            self.notify(text)
        else:
            print(text, flush=True)

    def current_settings(self):
        return copy.deepcopy(policy.execution_settings(self.net))

    def current_receipt(self):
        result = dict(copy.deepcopy(self._current), settings=self.current_settings())
        if self._cache is not None:
            result.update(cache_context_sha256=self._cache.context_sha256, audit_path=str(self.path))
        return result

    def restore_original(self):
        policy.apply_execution_settings(self.net, self.original)
        self._current = dict(reason="original_explicit_restore", settings=copy.deepcopy(self.original))

    def _apply(self, settings, reason, signature):
        policy.apply_execution_settings(self.net, settings)
        self._current = dict(reason=reason, workload_sha256=signature, settings=copy.deepcopy(settings))

    @staticmethod
    def _covered(inventory, envelope):
        return set(inventory) == set(envelope) and all(inventory[key] <= envelope[key] for key in inventory)

    def ensure(self, batch, workload):
        inventory = workload_inventory(workload)
        signature = _signature(inventory)
        if self._loaded_cache_pending:
            self._record('context_bound_policy_cache_loaded', measured_workloads=len(self._measured),
                         rejected_workloads=len(self._rejected), sidecar=str(self._cache.path))
            self._loaded_cache_pending = False
        if signature in self._uncached_original_only:
            self._apply(self.original, 'original_without_durable_measurement', signature)
            return None
        if signature in self._baseline_only:
            self._apply(self.original, "original_after_optimized_OOM", signature)
            return None
        for measured in reversed(self._measured):
            if self._covered(inventory, measured["inventory"]):
                self._apply(measured["selected"], "measured_inventory_heuristic", signature)
                return None

        # A small new high-water observation is admitted only with the original
        # execution. Doubling triggers another actual full-batch measurement;
        # this controls probe overhead, never graph/data/training coverage.
        high_water = {key: max(row["inventory"].get(key, 0) for row in self._measured)
                      for key in inventory} if self._measured else {}
        recalibrate = not self._measured or any(
            value > high_water[key] and value >= 2 * max(1, high_water[key])
            for key, value in inventory.items())
        if not recalibrate:
            self._apply(self.original, "original_for_unmeasured_larger_workload", signature)
            if signature not in self._unknown_reported:
                row = self._record("unmeasured_larger_workload_original_execution",
                                   workload_sha256=signature, inventory=inventory,
                                   settings=self.original, next_probe_growth_factor=2)
                if not self._unknown_notified:
                    self._notice("unmeasured_larger_workload_original_execution",
                                 workload_sha256=signature, settings=self.original)
                    self._unknown_notified = True
                self._unknown_reported.add(signature)
                return row
            return None

        self.restore_original()
        # Check this actual audit destination before expensive GPU measurement.
        # A successful small append cannot guarantee space for later receipts.
        self._record("calibration_started", workload_sha256=signature, inventory=inventory,
                     complete_original_batch=True, new_high_water=bool(self._measured))
        self._notice("calibration_started", workload_sha256=signature,
                     source_problems=inventory["source_problems"],
                     complete_original_batch=True, new_high_water=bool(self._measured))
        try:
            selected, receipt = policy.calibrate_execution_policy(
                self.net, batch, self.loss_fn, amp=self.amp,
                cuda_limit_bytes=self.cuda_limit_bytes, reserve_bytes=self.reserve_bytes,
                notify=self.notify)
        except Exception as error:
            self.restore_original()
            try:
                self._record("calibration_failed", workload_sha256=signature, inventory=inventory,
                             error=f"{type(error).__name__}: {error}",
                             candidate_reports=getattr(error, "reports", None))
            except Exception as audit_error:
                report_secondary_failure(error, audit_error, operation='calibration failure audit')
            raise
        row = self._record("calibration_completed", workload_sha256=signature,
                           inventory=inventory, selected=selected, calibration=receipt)
        measured = dict(inventory=inventory, selected=copy.deepcopy(selected))
        if self._cache is not None:
            try:
                measured = measured_entry(inventory, selected, receipt, self._cache.binding)
            except PolicyCacheError as error:
                # Retaining original execution requires no new optimization
                # admission. A partial no-headroom proof is never persisted,
                # but must not turn an otherwise valid baseline into a failure.
                if (receipt.get('calibration_status') != 'original_retained_no_headroom'
                        or selected != self.original):
                    raise
                self._record('original_policy_cache_admission_skipped', workload_sha256=signature,
                             reason=str(error), selected=self.original,
                             scope='original execution retained; incomplete evidence not persisted')
                self._uncached_original_only.add(signature)
                self._apply(self.original, 'original_without_durable_measurement', signature)
                self._notice('calibration_completed', workload_sha256=signature, settings=self.original)
                return row
            self._cache.publish([*self._measured, measured], self._rejected)
        self._measured.append(measured)
        self._apply(selected, "measured_current_actual_batch", signature)
        self._notice("calibration_completed", workload_sha256=signature, settings=selected)
        return row

    def reject_optimized_oom(self, workload, error):
        """Return whether a same-batch original-execution retry is available.

        The caller must invoke this only before an optimizer update and restore
        the attempt's RNG and gradients itself. Original-policy OOM is fatal.
        """
        current = self.current_settings()
        if current == self.original:
            return False
        inventory = workload_inventory(workload)
        signature = _signature(inventory)
        self._baseline_only.add(signature)
        self._apply(self.original, "original_after_optimized_OOM", signature)
        if self._cache is not None:
            rejected = dict(workload_sha256=signature, inventory=inventory)
            self._cache.publish(self._measured, [*self._rejected, rejected])
            self._rejected.append(rejected)
        self._record("optimized_execution_OOM_original_retry", workload_sha256=signature,
                     rejected=current, restored=self.original, error=f"{type(error).__name__}: {error}",
                     retry_scope="same complete batch before optimizer update; caller restores RNG and clears gradients")
        self._notice("optimized_execution_OOM_original_retry", workload_sha256=signature,
                     settings=self.original)
        return True
