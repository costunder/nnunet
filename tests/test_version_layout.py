"""Navigation/dispatch regressions; no medical data or training is executed."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('version_dispatch',ROOT/'versions/run.py')
dispatch=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatch)


class VersionLayoutTests(unittest.TestCase):
    def test_registry_targets(self):
        rows=dispatch.entries()
        self.assertEqual(len(rows),len({row['id'] for row in rows}))
        for row in rows:
            for value in row['configs']+row['modules']+row['notes']+([row['runner']] if row['runner'] else []):
                path=(ROOT/value).resolve(strict=True)
                self.assertTrue(path.is_relative_to(ROOT))

    def test_fixed_arm_preserves_original_arguments(self):
        supplied=['--gpu','3','--output','work/my_run','--workers','16']
        self.assertEqual(dispatch.arguments(dispatch.resolve('v1.7/D'),supplied),['--arm','D',*supplied])
        self.assertEqual(supplied,['--gpu','3','--output','work/my_run','--workers','16'])

    def test_conflicting_arm_rejected(self):
        for args in (['--arm','C'],['--arm=C'],['--arm','D','--arm','D']):
            with self.assertRaises(ValueError): dispatch.arguments(dispatch.resolve('v1.7/D'),args)

    def test_curriculum_option_is_explicit(self):
        args=dispatch.arguments(dispatch.resolve('v2.2/curriculum'),['--margin-mm','10'])
        self.assertEqual(args[:2],['--curriculum-config','config/v22_cumulative_u16.json'])

    def test_delegation_uses_original_filename_and_restores_process(self):
        observed=[];before_argv=sys.argv;before_path=list(sys.path);before_cwd=Path.cwd()
        def fake(path,run_name):
            observed.append((path,list(sys.argv),Path.cwd(),run_name))
            sys.path.insert(0,'original-script-added-path')
        with patch.object(dispatch.runpy,'run_path',fake): dispatch.launch('v1.7/D',['--gpu','3'])
        self.assertEqual(observed,[(str((ROOT/'tools/run_v17_crossed_training.py').resolve()),
                                   [str((ROOT/'tools/run_v17_crossed_training.py').resolve()),'--arm','D','--gpu','3'],ROOT,'__main__')])
        self.assertIs(sys.argv,before_argv);self.assertEqual(sys.path,before_path);self.assertEqual(Path.cwd(),before_cwd)

    def test_describe_does_not_import_runtime(self):
        with patch.object(dispatch.runpy,'run_path',side_effect=AssertionError('runtime imported')):
            with patch('builtins.print') as printer: dispatch.launch('v2.2/cnn',['--describe'])
        self.assertEqual(json.loads(printer.call_args.args[0])['runner'],'tools/run_local_cnn_experiment.py')

    def test_retired_version_does_not_start_a_fallback(self):
        with self.assertRaises(ValueError): dispatch.launch('v2',[])
        with self.assertRaises(ValueError): dispatch.resolve('v1.99')

    def test_short_aliases_reach_repo(self):
        for row in dispatch.entries():
            if not row['runner']: continue
            name='eval.py' if row['id'].endswith('/eval') else 'run.py'
            path=ROOT/'versions'/row['folder']/name
            observed=[]
            # Execute only the thin alias with a mocked dispatcher, not its runtime.
            with patch('runpy.run_path',return_value={'launch':lambda *args:observed.append(args)}) as helper:
                exec(compile(path.read_text(encoding='utf8'),str(path),'exec'),{'__file__':str(path),'__name__':'__main__'})
            self.assertEqual(Path(helper.call_args.args[0]),ROOT/'versions/run.py')
            self.assertEqual(observed[0][0],row['id'])

    def test_archive_move_proof(self):
        data=json.loads((ROOT/'validation/version_layout/history.json').read_text(encoding='utf8'))
        self.assertEqual(data['payload_mismatches'],0)
        self.assertEqual(data['directory_inventory_mismatches'],0)
        self.assertTrue(data['preserved_source_files_identical'])

    def test_version_markdown_local_links(self):
        documents=[ROOT/'START.md']+list((ROOT/'versions').rglob('*.md'))
        # Historic immutable reports may contain old paths; only new navigation is checked.
        for path in documents:
            if '/history/' in path.as_posix() and path.name!='README.md': continue
            if path.name=='history.md' or path.name not in ('README.md','code.md','config.md','files.md','runs.md','artifacts.md','START.md'): continue
            text=path.read_text(encoding='utf8')
            for match in re.finditer(r'\[[^\]]*\]\(([^\)]+)\)',text):
                target=match.group(1).split('#',1)[0]
                if not target or '://' in target or target.startswith('<'): continue
                self.assertTrue((path.parent/target).exists(),f'{path.relative_to(ROOT)}: {target}')


if __name__=='__main__':
    unittest.main()
