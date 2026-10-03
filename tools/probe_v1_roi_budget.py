"""Read-only bounded actual-CT CPU ROI cost probe; never starts training."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from hiercp_v1x.roi_budget_probe import run_probe, worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-request', help=argparse.SUPPRESS)
    p.add_argument('--experiment')
    p.add_argument('--output')
    p.add_argument('--candidate-voxel-budget', type=int)
    p.add_argument('--rss-gib', type=float)
    p.add_argument('--case-timeout-seconds', type=float)
    p.add_argument('--reuse-report', help='Verify and reuse completed source ROI costs; failed target rows are replayed')
    a = p.parse_args()
    if a.worker_request:
        worker(a.worker_request)
        return
    if any(value is None for value in (a.experiment, a.output, a.candidate_voxel_budget,
                                      a.rss_gib, a.case_timeout_seconds)):
        p.error('--experiment, --output, --candidate-voxel-budget, --rss-gib and --case-timeout-seconds are required')
    import math
    if not math.isfinite(a.rss_gib) or a.rss_gib <= 0:
        p.error('--rss-gib must be finite and positive')
    report = run_probe(a.experiment, a.output, candidate_voxels=a.candidate_voxel_budget,
                       rss_bytes=int(a.rss_gib*2**30), case_timeout_seconds=a.case_timeout_seconds,
                       reuse_report=a.reuse_report)
    print(json.dumps(dict(report=str(Path(a.output).resolve()),
        completed=report['completed'], failed_sample_requests=report['failed_sample_requests'],
        reused_source_measurements=report['reused_source_measurements'],
        new_measurements=report['new_measurements'],
        proposed_guard=report['candidate_roi_max_voxels'],
        peak_rss_gib=max(row['peak_rss_bytes'] for row in report['measurements'])/2**30,
        scope=report['scope'], originals_preserved=report['originals_preserved'],
        training_started=False, production_ready=False), ensure_ascii=False, indent=2), flush=True)
    if not report['completed']:
        raise RuntimeError(f"CPU geometry probe failed; all diagnostics preserved in {a.output}. No training/cache/config changes.")


if __name__ == '__main__':
    main()
