"""Actual CT/CUDA DEBUG of independent execution; never a quality evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from tools.run_v18_independent import validate_reference
    manifest = validate_reference(a.reference)
    if manifest['debug'] is not True:
        raise ValueError('Only an already signed actual-CT DEBUG reference is allowed')
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    originals = [a.reference/'experiment.json', a.reference/'initial.pt', a.reference/'calibration.json']
    originals += list(a.reference.glob('*/checkpoint*.pt'))
    before = {str(path): sha(path) for path in originals}
    reference = output/'fresh_reference'
    reference.mkdir()
    # Common measured neural init/calibration are real preserved DEBUG evidence.
    # Do not copy a completed arm: this DEBUG exercises new actual updates.
    for name in ['experiment.json', 'initial.pt', 'calibration.json']:
        shutil.copyfile(a.reference/name, reference/name)
    for path in a.reference.glob('calibration_*.json'):
        shutil.copyfile(path, reference/path.name)
    command = [sys.executable, '-B', '-u', str(ROOT/'tools/run_v18_independent.py'),
               '--gpu', str(a.gpu), '--arm', 'native', '--source-experiment', str(reference),
               '--experiment', str(output/'native'), '--inventory', str(a.inventory.resolve()),
               '--debug-bank', str(a.bank.resolve())]
    for invocation in ('actual_updates', 'completed_resume'):
        if invocation == 'completed_resume':
            stable = {str(path): sha(path) for path in (output/'native/native').glob('checkpoint*.pt')}
        with (output/(invocation+'.log')).open('x', encoding='utf8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if invocation == 'completed_resume' and any(sha(Path(path)) != checksum for path, checksum in stable.items()):
            raise RuntimeError('Completed resume changed saved checkpoint bytes')
    after = {str(path): sha(path) for path in originals}
    if after != before:
        raise RuntimeError('Original DEBUG experiment bytes changed')
    run = output/'native/native'
    contract = json.loads((run/'execution_contract.json').read_text())
    rows = [json.loads(line) for line in (run/'update_timing.jsonl').read_text().splitlines()]
    complete = json.loads((run/'training_complete.json').read_text())
    reports = [json.loads(path.read_text()) for path in sorted(run.glob('validation_epoch_*.json'))]
    if contract['parameters'] != 10434532 or len(rows) != manifest['epochs']:
        raise RuntimeError('Full original DEBUG model/update contract differs')
    report = dict(scope='actual CT/CUDA DEBUG; not server192GiB or ranking-quality validation',
        original_preserved=before == after, model_parameters=contract['parameters'],
        execution_contract=contract, actual_updates=rows, completion=complete,
        whole129_validation_reports=reports, completed_resume_checkpoint_bytes_unchanged=True,
        completed_resume_log_sha256=sha(output/'completed_resume.log'),
        full_server_training=False, quality_verified=False,
        Linux_madvise_malloc_trim_tested=False)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print('ACTUAL CT/CUDA DEBUG REPORT: ' + str(output/'report.json'), flush=True)


if __name__ == '__main__':
    main()
