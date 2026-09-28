"""Foreground full-cohort preparation only; never starts model training.

Calls the existing observation, disk-admission and graph preparation functions.
The reference cache is used for size estimates only, never for graph reuse.
"""
import argparse
import ast
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


class ProgressLog:
    """Keep complete stdout in the log and render existing structured events."""
    def __init__(self, log, bar):
        from tools.watch_v222_server import Progress
        self.log, self.bar = log, bar
        self.state = Progress()
        self.partial = ''
        self.lock = threading.RLock()
        self.last_refresh = 0.
        self.display_phase = None

    def stage(self, name):
        with self.lock:
            self.state.phase_to(name)
            self.bar.set_description_str(name, refresh=False)
            self.bar.total = None
            self.bar.reset()
            self.display_phase = name
            self.bar.set_postfix_str('', refresh=False)
            self.bar.refresh()

    def write(self, value):
        with self.lock:
            self.log.write(value)
            self.log.flush()
            self.partial += value
            while '\n' in self.partial:
                line, self.partial = self.partial.split('\n', 1)
                # Human-readable lines remain in console.log. Only JSON event
                # lines drive the display; malformed events fail explicitly.
                if not line.startswith('{'):
                    continue
                event = ast.literal_eval(line) if line.startswith("{'stage':") else json.loads(line)
                if not isinstance(event, dict):
                    continue
                self.state.event(event)
                self.bar.set_description_str(self.state.phase, refresh=False)
                self.bar.total = self.state.total
                if self.display_phase != self.state.phase:
                    self.bar.reset()
                    self.display_phase = self.state.phase
                self.bar.update(self.state.done - self.bar.n)
                self.bar.set_postfix_str(self.state.detail, refresh=False)
                if time.monotonic() - self.last_refresh >= .2:
                    self.bar.refresh()
                    self.last_refresh = time.monotonic()
            return len(value)

    def flush(self):
        with self.lock:
            self.log.flush()


def prepare_only(medical, reference, output, progress, announce):
    from tools.v1_server import observations
    from tools.v22_cache_storage import make_plan, admit
    from tools.v222_prepare_optimized import prepare
    from hiercp_v222.v1_cache import configuration
    cfg, _ = configuration()
    progress.stage('observations: full cohort')
    observations(medical, ROOT/'config/split_cp80_fold0.json', output/'observations')
    index = output/'observations/index.json'
    target = output/'paired_cache'
    progress.stage('storage admission')
    plan = make_plan(reference, index, target)
    write_new(output/'storage_plan.json', plan)
    admission = admit(plan, index, target, plan['counts'], int(cfg['minimum_free_gb']*1024**3))
    write_new(output/'storage_checked.json', admission)
    announce('Storage admitted: estimated %.2f GiB, free %.2f GiB, reserve %.2f GiB. '
             'Estimate is not a guaranteed upper bound.' % (
                 admission['projected_payload_bytes']/1024**3,
                 admission['free_before_bytes']/1024**3,
                 admission['reserved_free_bytes']/1024**3))
    progress.stage('paired cache: full cohort')
    prepared = prepare(index, target, storage_plan=output/'storage_plan.json')
    # The existing writer checks cohort completeness. Record completion only
    # after its return and final index; no training entry point is invoked.
    meta = json.loads(Path(prepared).read_text(encoding='utf-8-sig'))
    if meta.get('complete') is not True or meta.get('debug') is not False:
        raise RuntimeError('Preparation returned without a complete production index')
    result = dict(cache=str(prepared), records=len(meta['records']), training_started=False,
                  G3_passed=False, G4_passed=False, scope='full-cache preparation only')
    write_new(output/'preparation_complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--medical-root', type=Path, required=True)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    medical, reference, output = (p.resolve() for p in (args.medical_root,args.reference_cache,args.output))
    if not reference.is_file():raise FileNotFoundError(reference)
    for folder in ('image','labels'):
        if not (medical/'Data'/folder).is_dir():raise FileNotFoundError(medical/'Data'/folder)
    from tqdm import tqdm
    from hiercp.preparation_runtime import snapshot
    from hiercp_v222.v1_cache import configuration, provenance
    cfg, base = configuration()
    allocation = snapshot()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/'requested.json', dict(medical_root=str(medical),reference_cache=str(reference),
        source_identity=provenance(),config=cfg,base=base,resources=allocation,
        training_started=False,reference_use='file-size estimate only',subset=False))
    print(f'Preparation only | {output}\nLog: {output / "console.log"}', flush=True)
    print('Ctrl+C interrupts foreground preparation; active CPU tasks may need time to unwind. '
          'Existing outputs are preserved. No detached training worker.', flush=True)
    with (output/'console.log').open('x', encoding='utf-8', newline='\n') as log:
        with tqdm(total=None, unit='item', dynamic_ncols=True, file=sys.stderr) as bar:
            progress = ProgressLog(log, bar)
            try:
                with redirect_stdout(progress):
                    result = prepare_only(medical, reference, output, progress,
                                          lambda message:tqdm.write(message, file=sys.stderr))
            except KeyboardInterrupt:
                write_new(output/'preparation_interrupted.json', dict(training_started=False, complete=False))
                print('Preparation interrupted; outputs preserved. No training started.', file=sys.stderr)
                return 130
            except Exception as error:
                write_new(output/'preparation_failed.json', dict(type=type(error).__name__,error=str(error),
                          training_started=False,complete=False))
                raise
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
