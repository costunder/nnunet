"""Read-only inspection of historical/current population banks, not migration.

The optional JSON report is published without replacing any existing file.
Structural validity of a legacy bank is not proof of its fitted memberships or
permission to load it into the current GNN.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from hiercp.prototype import audit_prototype_bank


def publish_report(report: dict, destination: Path) -> None:
    """Atomically create a report; never overwrite historical evidence."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Preserve the existing report; choose a NEW output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                        dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory link is a no-clobber atomic publication, including
        # a race with another writer after the initial existence check.
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-bank", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        help="Optional NEW JSON report path; without it only stdout is written")
    args = parser.parse_args(argv)
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f"Existing output is preserved: {args.output}")
    report = audit_prototype_bank(args.prototype_bank)
    if args.output is not None:
        publish_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
