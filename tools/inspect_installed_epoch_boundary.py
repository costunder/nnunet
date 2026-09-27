"""Read installed trainer source/MRO; never construct a trainer or send signals."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_record(function):
    lines, first = inspect.getsourcelines(function)
    source = ''.join(lines)
    path = Path(inspect.getfile(function)).resolve()
    tree = ast.parse(textwrap.dedent(source))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    return dict(path=str(path), first_line=first, source=source,
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                direct_exit_calls=[c for c in calls if c in
                                   ('exit', 'quit', 'sys.exit', 'os._exit', 'SystemExit')],
                calls=calls)


def inspect_boundary():
    import torch
    from custom_trainers.nnUNetTrainer_OnlineRankV22 import nnUNetTrainer_250epochs_OnlineRankV22
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    trainer = nnUNetTrainer_250epochs_OnlineRankV22
    overrides = []
    for cls in trainer.__mro__:
        if 'on_epoch_end' in cls.__dict__:
            overrides.append(dict(owner=f'{cls.__module__}.{cls.__name__}',
                                  **source_record(cls.__dict__['on_epoch_end'])))
    return dict(read_only=True, trainer_constructed=False, signals_sent=False,
                CUDA_initialized=torch.cuda.is_initialized(),
                mro=[f'{c.__module__}.{c.__name__}' for c in trainer.__mro__],
                epoch_overrides=overrides,
                installed_methods={n:source_record(getattr(nnUNetTrainer,n)) for n in
                                   ('on_epoch_end','on_train_end','run_training','save_checkpoint')},
                interpretation='Source excerpts and direct calls only; not proof that indirect calls cannot exit. '
                               'Inspect the actual server environment separately. No compatibility or training approval.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=inspect_boundary()
    encoded=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:
        with args.output.open('x',encoding='utf-8',newline='\n') as stream:stream.write(encoded+'\n')
        print(json.dumps(dict(output=str(args.output),CUDA_initialized=report['CUDA_initialized'],
                             epoch_owners=[r['owner'] for r in report['epoch_overrides']],
                             direct_exit_calls=[r['direct_exit_calls'] for r in report['epoch_overrides']]),indent=2))
    else:print(encoded)


if __name__=='__main__':main()
