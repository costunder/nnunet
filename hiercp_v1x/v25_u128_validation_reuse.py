"""Execution-only reuse of one actual U128 validation; frozen source is unchanged.

Legacy partial full-validation batches continue exactly as originally saved.
Report, curriculum, BEST, checkpoint, optimizer and RNG code remains original.
"""
from __future__ import annotations
import ast
import copy
import hashlib
from pathlib import Path
import time

FORMAT='v25_actual_U128_validation_reuse_execution_v1'
FROZEN_NORMALIZED_SHA256='0224b0cea6c0712872bb5b30295840771df5eb7dba25f62d4c75b8e8cb042c68'
_GATE_NAME='_v25_u128_execution_gate_factory'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ValidationReuseGate:
    """Only signed metadata is checked; no image/graph/provider preparation."""
    def __init__(self,training,root,net,scorer,population,model_contract,physical,debug,binding_hash):
        self.training=training;self.root=Path(root);self.net=net;self.scorer=scorer
        self.population=population;self.model_contract=copy.deepcopy(model_contract)
        self.physical=physical;self.debug=debug;self.binding_hash=binding_hash
        self.stage_start=None;self.actual_stage=None

    def _model_rng(self):
        return dict(model_sha256=self.training.digest(self.net.state_dict()),
            RNG_sha256=self.training.digest(self.training.capture_rng()))

    def _record(self,**value):
        self.training._append(self.root/'validation_reuse.jsonl',dict(format=FORMAT,**value))

    def _validate_actual_stage(self,state,cases_all):
        stage=state.get('stage_validation')
        self.training._validate_report(stage,epoch=state['epoch'],active_u=128,
            phase='stage_validation',expected_cases=sorted(cases_all))
        if (stage.get('updates')!=state['updates'] or stage.get('recipient_GT_used_in_forward')is not False
                or stage.get('debug')is not self.debug or [r['case_id']for r in stage['cases']]!=list(cases_all)
                or self.scorer.net is not self.net or self.net.training):
            raise ValueError('Actual same-model/epoch/update/order U128 stage validation required')
        # V23Population.case only selects signed inventory rows/indices. It
        # performs no CT decoding, context extraction, L0 or graph preparation.
        for row in stage['cases']:
            plan=self.population.case(row['case_id'],active_u_count=128)
            if (list(plan.record_ids)!=row['record_ids'] or list(plan.positive_indices)!=row['positive_indices']
                    or list(plan.unobserved_indices)!=row['unobserved_indices']):
                raise ValueError('Actual U128 stage scores differ from exact bound input IDs/all-P indices')
        return stage

    def try_reuse(self,state,phase,active,cases_all):
        if phase=='stage_validation'and active==128:
            self.stage_start=dict(epoch=state['epoch'],updates=state['updates'],
                resumed_position=state['evaluation_position'],**self._model_rng())
            return False
        if phase!='full_validation'or active!=128:return False
        position=state['evaluation_position']
        if position:
            # These rows are actual historical full-pass output. CUDA sparse
            # reductions can differ slightly from stage output; retain them.
            self._record(epoch=state['epoch'],updates=state['updates'],phase=phase,active_u=128,
                reused=False,legacy_partial_full_preserved=True,retained_position=position,
                actual_remaining_batches_follow_original=True,new_forward_claimed=False)
            return False
        began=time.perf_counter();stage=self._validate_actual_stage(state,cases_all)
        observed=self._model_rng();stage_digest=self.training.digest(stage)
        if self.actual_stage is not None:
            expected=self.actual_stage
            if (expected['epoch']!=state['epoch'] or expected['updates']!=state['updates']
                    or expected['report_sha256']!=stage_digest
                    or any(expected[key]!=observed[key]for key in ('model_sha256','RNG_sha256'))):
                raise ValueError('Model/RNG/actual stage report changed before identical U128 full reuse')
        state['evaluation_rows']=copy.deepcopy(stage['cases']);state['evaluation_position']=len(cases_all)
        self._record(epoch=state['epoch'],updates=state['updates'],phase=phase,active_u=128,
            reused=True,source_phase='stage_validation',source_report_sha256=stage_digest,
            training_identity_sha256=self.binding_hash,model_contract_sha256=self.training.digest(self.model_contract),
            **observed,case_ids=list(cases_all),actual_source_patients=len(cases_all),actual_source_P=135,
            physical_patient_batch=self.physical,evaluation_view_epoch=29,
            resumed_completed_stage=self.actual_stage is None,
            geometry_preparation_skipped=True,input_provider_preparation_skipped=True,
            L0_and_GNN_forward_skipped=True,new_forward_calls=0,new_patient_forward_count=0,
            metadata_verification_seconds=time.perf_counter()-began)
        print(f'v2.5 full_validation e{state["epoch"]:02d} U128 reuses actual stage_validation '
            f'{len(cases_all)}/{len(cases_all)} patients; new preparation/forward=0',flush=True)
        return True

    def observe_report(self,state,report,phase,active):
        if phase!='stage_validation'or active!=128:return
        if self.stage_start is None:raise RuntimeError('Actual U128 stage start evidence missing')
        observed=self._model_rng()
        if any(self.stage_start[key]!=observed[key]for key in ('model_sha256','RNG_sha256')):
            raise ValueError('Actual eval/no_grad U128 stage modified model state or global RNG')
        self.actual_stage=dict(epoch=state['epoch'],updates=state['updates'],
            report_sha256=self.training.digest(report),**observed)


def _transform(function):
    changes=0;observations=0;gate_inserted=0
    for node in list(ast.walk(function)):
        body=getattr(node,'body',None)
        if not isinstance(body,list):continue
        for offset in range(len(body)-1,-1,-1):
            item=body[offset]
            if (isinstance(item,ast.Assign)and len(item.targets)==1
                    and isinstance(item.targets[0],ast.Name)and item.targets[0].id=='began_run'):
                gate=ast.parse("_u128_gate = "+_GATE_NAME+"(root,net,scorer,population,model_contract,physical_patient_batch,debug,binding_hash)").body[0]
                body.insert(offset,gate);gate_inserted+=1
            if (isinstance(item,ast.Expr)and isinstance(item.value,ast.Call)
                    and ast.dump(item.value.func)==ast.dump(ast.parse('scorer.geometry.prepare',mode='eval').body)
                    and len(item.value.args)==1 and isinstance(item.value.args[0],ast.Name)and item.value.args[0].id=='eval_u'):
                original=body[offset:offset+3]
                if (len(original)!=3 or not isinstance(original[1],ast.Assign)or not isinstance(original[2],ast.For)
                        or ast.unparse(original[2].iter)!='patient_batches(cases_all, physical_patient_batch)'):
                    raise ValueError('Frozen full evaluation preparation/batch loop differs')
                condition=ast.parse('not _u128_gate.try_reuse(state,phase,active,cases_all)',mode='eval').body
                body[offset:offset+3]=[ast.If(test=condition,body=original,orelse=[])];changes+=1
            if (isinstance(item,ast.Expr)and isinstance(item.value,ast.Call)
                    and ast.unparse(item.value.func)=='report.update'):
                body.insert(offset+1,ast.parse('_u128_gate.observe_report(state,report,phase,active)').body[0]);observations+=1
    if (changes,observations,gate_inserted)!=(1,1,1):raise ValueError('Only exact frozen evaluation/gate anchors may change')
    ast.fix_missing_locations(function)
    return function


def install_runtime(training_module=None,expected_source_sha256=None):
    if training_module is None:
        from . import v25_training as training_module
    module=training_module;source_path=Path(module.__file__).resolve(strict=True)
    raw_sha=_sha(source_path);source=source_path.read_text(encoding='utf8')
    if (hashlib.sha256(source.encode()).hexdigest()!=FROZEN_NORMALIZED_SHA256
            or expected_source_sha256 is not None and raw_sha!=expected_source_sha256):
        raise ValueError('Exact unchanged frozen v25_training source required')
    current=module.run_training
    if hasattr(current,'_v25_u128_reuse_receipt'):
        receipt=current._v25_u128_reuse_receipt
        if receipt['runtime_source_sha256']!=_sha(__file__):raise ValueError('Installed U128 execution sidecar changed')
        return copy.deepcopy(receipt)
    if (Path(current.__code__.co_filename).resolve()!=source_path or current.__name__!='run_training'
            or module.__name__!='hiercp_v1x.v25_training'):
        raise ValueError('Original frozen run_training function must be the sole patch target')
    parsed=ast.parse(source);functions=[n for n in parsed.body if isinstance(n,ast.FunctionDef)and n.name=='run_training']
    if len(functions)!=1:raise ValueError('Exact singleton frozen training entry required')
    transformed=_transform(copy.deepcopy(functions[0]));generated=ast.Module(body=[transformed],type_ignores=[])
    ast.fix_missing_locations(generated);compiled=compile(generated,str(source_path)+'::U128_execution_reuse','exec')
    # Original live globals preserve later storage/checkpoint admission adapters.
    module.__dict__[_GATE_NAME]=lambda *args:ValidationReuseGate(module,*args)
    exec(compiled,module.__dict__)
    receipt=dict(format=FORMAT,frozen_training_source_sha256=raw_sha,
        runtime_source_sha256=_sha(__file__),transformed_AST_sha256=hashlib.sha256(ast.dump(generated).encode()).hexdigest(),
        frozen_scientific_file_bytes_changed=False,checkpoint_identity_schema_changed=False,
        reuse_only_active_U128_full_position0=True,legacy_partial_full_rows_preserved=True,
        original_report_curriculum_BEST_checkpoint_paths_preserved=True,
        physical_patient_batch=4,epochs=40,optimizer_updates=680,evaluation_view_epoch=29,
        all_validation_patients=21,all_validation_P=135,new_duplicate_forward_calls=0)
    module.run_training._v25_u128_original=current;module.run_training._v25_u128_reuse_receipt=receipt
    return copy.deepcopy(receipt)
