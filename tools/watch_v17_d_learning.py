"""Display all existing D validation epochs and optionally follow new records.

Read-only: no GPU selection, checkpoint load, forward, optimizer or new evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--follow', action='store_true', help='Follow new saved metrics; Ctrl+C stops only this display')
    args = parser.parse_args(argv)
    from hiercp_v1x.transition_learning_display import DLearningDisplay
    display = DLearningDisplay(args.experiment)
    print('D SAVED LEARNING | ' + str(display.root), flush=True)
    print(display.scope_line(), flush=True)
    print('Hit@1=case first rank is observed P; R@1=observed-P micro recall.', flush=True)
    print('Train loss/gradients show update activity. Ranking improvement is measured by the validation history below.', flush=True)
    last_progress = 0.
    first = True
    try:
        while True:
            lines, changed = display.poll()
            for line in lines:
                print(line, flush=True)
            now = time.monotonic()
            if first or lines or changed and now - last_progress >= 30:
                print(display.progress_line(), flush=True)
                print(display.delta_line(), flush=True)
                last_progress = now
            if first and display.curve.pending:
                print('Latest validation JSONL line is still being written; it will be shown when complete.', flush=True)
            complete = display.completion()
            if complete is not None:
                print('D COMPLETE | epochs=40 | steps=' + str(complete.get('steps'))
                      + ' | selected BEST epoch=' + str(complete.get('selected_epoch')), flush=True)
                break
            if not args.follow:
                if not display.history:
                    print('No complete validation record yet; no ranking result substituted.', flush=True)
                break
            if first:
                print('Following saved records. No new validation line during support refresh/evaluation is expected. Ctrl+C stops only this viewer.', flush=True)
            first = False
            time.sleep(5)
    except KeyboardInterrupt:
        print('\nDisplay stopped; D training and checkpoint are unchanged.', flush=True)


if __name__ == '__main__':
    main()
