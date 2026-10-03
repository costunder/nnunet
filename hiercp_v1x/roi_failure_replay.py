"""Capture the first failed ROI call from the unmodified native sample builder.

This diagnostic executes the original source selection, candidate search,
curriculum, canonical coordinates and graph materialisation. It never returns a
training sample or substitutes a graph. Temporary call observers are restored
even when replay fails. Returned arrays and objects are in-process call inputs,
not JSON evidence; the caller owns the separately bounded cost measurement.
"""
from __future__ import annotations

import inspect
import math


def _spec_metadata(spec):
    result = {
        "center": [int(v) for v in spec.center],
        "difficulty": int(spec.difficulty),
        "corruption": int(spec.corruption),
        "scale_xyz": [float(v) for v in spec.scale_xyz],
        "rotation": [float(v) for v in spec.rotation],
    }
    if (len(result["center"]) != 3 or len(result["scale_xyz"]) != 3
            or len(result["rotation"]) != 9
            or not all(math.isfinite(v) for key in ("scale_xyz", "rotation")
                       for v in result[key])):
        raise ValueError("Original replay candidate transform metadata is invalid")
    for key in ("region_id", "prototype_id"):
        if hasattr(spec, key):
            result[key] = int(getattr(spec, key))
    return result


def capture_failed_roi(cache, local, spatial, *, sample_kwargs,
                       expected_geometry, parse_failure):
    """Replay the complete original sample until its first ROI budget error.

    ``sample_kwargs`` is forwarded unchanged to ``cache.build_training_sample``
    and must contain the original graph configuration. Only call observers are
    installed: no node/edge, source, candidate or geometry operation is bypassed.

    The returned dictionary contains ``operation``, ``original_failure``,
    ``replayed_geometry``, source identity, ``phase`` and plain ``target_spec``
    metadata. ``payload_kwargs`` or ``transform_args``/``transform_kwargs`` are
    the exact shallow call inputs. ``target_context`` holds the original case,
    CandidateSpec and target-preparation kwargs for a transform preallocation
    failure. No exception/traceback is retained, avoiding accidental retention
    of previously completed candidate graphs during the next cost phase.
    """
    if not isinstance(sample_kwargs, dict):
        raise TypeError("Original sample kwargs must be a dictionary")
    graph_config = sample_kwargs.get("graph_config")
    if (graph_config is None or
            graph_config.adaptive_roi_max_voxels != expected_geometry["voxel_budget"]):
        raise ValueError("Replay must use the exact failed original ROI budget")
    originals = {
        "source": cache.prepare_local_source,
        "target": local._prepare_local_target,
        "payload": local.build_patch_payload,
        "transform": local.transform_footprint_physical,
    }
    state = {"phase": None, "target": None, "source": None}
    captures = []

    def capture(operation, args, kwargs, error):
        # A nested observer may see the same propagated error. Preserve its
        # deepest operation, while the top-level error still must match it.
        if captures:
            return
        target = state["target"]
        source = state["source"]
        item = {
            "operation": operation,
            "original_failure": str(error),
            "phase": state["phase"],
            "source_component": None if source is None else source["component"],
            "source_anchor": None if source is None else source["anchor"],
            "target_spec": None if target is None else target["metadata"],
            "target_context": None if target is None else {
                "case": target["case"], "spec": target["spec"],
                "kwargs": dict(target["kwargs"]),
            },
            "payload_kwargs": dict(kwargs) if operation == "build_patch_payload" else None,
            "payload_args": tuple(args) if operation == "build_patch_payload" else None,
            "transform_args": tuple(args) if operation == "transform_footprint_physical" else None,
            "transform_kwargs": dict(kwargs) if operation == "transform_footprint_physical" else None,
        }
        captures.append(item)

    def source_observer(*args, **kwargs):
        bound = inspect.signature(originals["source"]).bind(*args, **kwargs)
        source = bound.arguments["source"]
        state["source"] = {
            "component": int(source.component_id),
            "anchor": [int(v) for v in source.anchor_center],
        }
        previous = state["phase"]
        state["phase"] = "source"
        try:
            return originals["source"](*args, **kwargs)
        finally:
            state["phase"] = previous

    def target_observer(*args, **kwargs):
        bound = inspect.signature(originals["target"]).bind(*args, **kwargs)
        case, spec = bound.arguments["case"], bound.arguments["spec"]
        previous_phase, previous_target = state["phase"], state["target"]
        state["phase"] = "target"
        state["target"] = {
            "case": case, "spec": spec, "metadata": _spec_metadata(spec),
            # The prepared source graph is unnecessary for finishing this ROI
            # call and can be very large. Its footprint is already captured by
            # the transform/payload observer; do not retain its node/edge tables.
            "kwargs": {k: v for k, v in bound.arguments.items()
                       if k not in ("case", "spec", "prepared_source")},
        }
        try:
            return originals["target"](*args, **kwargs)
        finally:
            state["phase"], state["target"] = previous_phase, previous_target

    def payload_observer(*args, **kwargs):
        try:
            return originals["payload"](*args, **kwargs)
        except spatial.AdaptiveRoiBudgetError as error:
            capture("build_patch_payload", args, kwargs, error)
            raise

    def transform_observer(*args, **kwargs):
        try:
            return originals["transform"](*args, **kwargs)
        except spatial.AdaptiveRoiBudgetError as error:
            capture("transform_footprint_physical", args, kwargs, error)
            raise

    cache.prepare_local_source = source_observer
    local._prepare_local_target = target_observer
    local.build_patch_payload = payload_observer
    local.transform_footprint_physical = transform_observer
    try:
        try:
            cache.build_training_sample(**sample_kwargs)
        except spatial.AdaptiveRoiBudgetError as error:
            replayed = parse_failure(str(error))
            if replayed != expected_geometry:
                raise ValueError("First original ROI failure differs from the failed manifest") from error
            if len(captures) != 1 or captures[0]["original_failure"] != str(error):
                raise ValueError("First original ROI failure was not captured at one actual ROI call") from error
            item = captures[0]
            if item["phase"] not in ("source", "target") or item["source_component"] is None:
                raise ValueError("Failed ROI call lacks its actual source/target replay context") from error
            if item["phase"] == "target" and item["target_spec"] is None:
                raise ValueError("Failed target ROI lacks its original curriculum specification") from error
            item["replayed_geometry"] = replayed
            return item
        raise ValueError("Original sample completed without reproducing the failed ROI guard")
    finally:
        cache.prepare_local_source = originals["source"]
        local._prepare_local_target = originals["target"]
        local.build_patch_payload = originals["payload"]
        local.transform_footprint_physical = originals["transform"]
