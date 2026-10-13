"""Actual frozen native argparse boundary, CPU only; no dataset/model execution."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT=Path(__file__).resolve().parents[1]
RUNTIME_SHA='114150eb157f92f460c5d6bc20ad77ae501ac58be03c4e0cd6fb7f5d133f5c3a'
PREFIX_SHA='c32e13550c20a6a45e7395a7331e3dd651730b35bb64432b48c819dc7bb3e617'


def parser():
    fixture=json.loads((ROOT/'tests/fixtures/basic_cp_actual_native_cli_parser_boundary_v1.json').read_text(encoding='utf8'))
    if (fixture['runtime_sha256']!=RUNTIME_SHA or fixture['parser_prefix_sha256']!=PREFIX_SHA
            or hashlib.sha256(fixture['parser_prefix'].encode()).hexdigest()!=PREFIX_SHA
            or fixture['source_metadata_unchanged'] is not True or fixture['contains_patient_data'] is not False):
        raise AssertionError('Exact actual readonly native CLI source fixture required')
    namespace={}
    exec(compile(fixture['parser_prefix'],'<actual native argparse declaration only>','exec'),namespace)
    return namespace['run_training_entry']()


def runner_argv(resumed):
    # Execute only actual runner argument-construction statements, never train().
    tree=ast.parse((ROOT/'tools/run_basic_cp_ablation.py').read_text(encoding='utf8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    def target(node):return isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='sys' and node.attr=='argv'
    statements=[]
    for node in function.body:
        if isinstance(node,ast.Assign) and any(target(t) for t in node.targets):statements.append(copy.deepcopy(node))
        if isinstance(node,ast.If) and any(isinstance(n,ast.AugAssign) and target(n.target) for n in ast.walk(node)):
            statements.append(copy.deepcopy(node))
    if len(statements)!=2:raise AssertionError('Unique actual base/resume CLI statements required')
    native={'trainer':'nnUNetTrainer_250epochs_BasicCPBlend','baseline':{'plans_name':'nnUNetResEncUNetMPlans'}}
    if resumed:native['completed_native_resume']={'completed_epochs':200}
    process=SimpleNamespace(argv=[])
    namespace={'sys':process,'native':native}
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),
                 '<actual runner argv statements only>','exec'),namespace)
    return process.argv


class ActualNativeCLIBoundaryTests(unittest.TestCase):
    def test_actual_runner_resumed_argv_parses_c_and_best_full_dataset730_fold0(self):
        argv=runner_argv(True)
        self.assertIn('--c',argv);self.assertNotIn('-c',argv)
        args=parser().parse_args(argv[1:])
        self.assertEqual((args.dataset_name_or_id,args.configuration,args.fold),('730','3d_fullres','0'))
        self.assertEqual(args.tr,'nnUNetTrainer_250epochs_BasicCPBlend')
        self.assertEqual(args.p,'nnUNetResEncUNetMPlans')
        self.assertTrue(args.c);self.assertTrue(args.val_best);self.assertFalse(args.val)
        self.assertIsNone(args.pretrained_weights);self.assertFalse(args.disable_checkpointing)
        self.assertEqual(args.device,'cuda');self.assertEqual(args.num_gpus,1)

    def test_actual_parser_rejects_failed_single_dash_token(self):
        argv=runner_argv(True)[1:];argv[argv.index('--c')]='-c'
        stderr=io.StringIO()
        with contextlib.redirect_stderr(stderr),self.assertRaises(SystemExit) as caught:
            parser().parse_args(argv)
        self.assertEqual(caught.exception.code,2)
        self.assertIn('unrecognized arguments: -c',stderr.getvalue())

    def test_actual_fresh_train_argv_keeps_best_without_resume_or_pretraining(self):
        args=parser().parse_args(runner_argv(False)[1:])
        self.assertFalse(args.c);self.assertTrue(args.val_best);self.assertIsNone(args.pretrained_weights)


if __name__=='__main__':unittest.main()
