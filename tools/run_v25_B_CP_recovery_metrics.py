"""Admit the actual original BEST26 evaluator without changing metric code."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from hiercp_v1x.v25_B_CP_recovery_runtime import install, ORIGINAL_METRICS_ENTRY
    install()
    from hiercp_v1x import v25_B_early_downstream_resources as previous
    seal = previous.load_seal()
    lease = previous.BEarlyDownstreamAllocationLease(seal['slot']['direct_proof']['path'])
    lease.verify(force=True)
    return runpy.run_path(str(ROOT/ORIGINAL_METRICS_ENTRY), run_name='__main__')


if __name__ == '__main__':
    main()
