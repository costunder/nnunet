"""Foreground v1 sampler bootstrap; does not start work unless called by CLI."""
from __future__ import annotations

import argparse
import functools
import importlib
import os
from pathlib import Path
import sys

from hiercp_v1x.sampling_runtime import ENVIRONMENT_KEY


def preserve_worker_calibration_rng(pipeline):
    """Worker auto-calibration must not alter subsequent model/dropout RNG.

    The fixed-worker comparator skips this probe. Restore all original v1 RNG
    states in either successful or failed calibration; model math is unchanged.
    """
    original = pipeline._measure_worker_candidates
    if getattr(original, "v1x_rng_preserved", False):
        return original
    from hiercp.tensor import capture_rng_state, restore_rng_state

    @functools.wraps(original)
    def measured(*args, **kwargs):
        state = capture_rng_state()
        try:
            return original(*args, **kwargs)
        finally:
            restore_rng_state(state)

    measured.v1x_rng_preserved = True
    pipeline._measure_worker_candidates = measured
    return measured


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bound native/strict-nested v1 foreground entry")
    parser.add_argument("--contract", required=True)
    args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if not remaining:
        parser.error("Original hiercp.pipeline arguments are required")
    path = Path(args.contract).resolve(strict=True)
    os.environ[ENVIRONMENT_KEY] = str(path)
    from hiercp_v1x.sampling_runtime import install_from_environment, load_environment_contract
    contract = load_environment_contract()
    # The strict snapshot sample hook executes during pipeline import and also
    # on worker imports. Native snapshots have no hook and are left untouched.
    pipeline = importlib.import_module("hiercp.pipeline")
    install_from_environment()
    preserve_worker_calibration_rng(pipeline)
    print(f"Sampling | mode={contract['mode']} | contract={contract['contract_sha256']} "
          "| worker calibration RNG restored | no learned-feature cache", flush=True)
    previous = sys.argv
    try:
        sys.argv = ["hiercp.pipeline", *remaining]
        return pipeline.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
