"""Original native250/BEST26 arguments with inherited recovery admission."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    from hiercp_v1x.v25_B_CP_recovery_runtime import install
    install()
    from tools.run_v25_authorized_early_downstream import main as original
    return original(argv)


if __name__ == '__main__':
    raise SystemExit(main())
