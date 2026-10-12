"""DEBUG CPU contract tests for independent complete prefix admission."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from hiercp_v1x.v25_independent_preparation import (
    FORMAT, _hash, install_prefix_lock, validate_shared_preparation, validate_arm_preparation)
from hiercp_v1x.v25_independent_preparation import publish_runtime_receipt
from hiercp_v1x.v25_targets import validate_policy


class IndependentPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='DEBUG_v25_independent_preparation_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.group = self.root/'group';self.group.mkdir()
        self.cache = self.root/'cache';(self.cache/'local').mkdir(parents=True)
        for arm in 'ABCD':(self.group/arm).mkdir()
        self.bindings = dict(format='v25_immutable_CT_organ_fixed_donor_shared_context_bindings_v1',
            case_ids=['DEBUG_case_'+str(i) for i in range(105)], complete=True)
        self.bindings['bindings']={case:{'DEBUG_binding':case} for case in self.bindings['case_ids']}
        fields={key:_hash({'DEBUG_binding_field':key}) for key in
            ('query_rows_sha256','config_sha256','source_sha256','prototype_bank_sha256','context_contract_sha256')}
        self.bindings.update(fields)
        self.bindings['content_sha256'] = _hash(self.bindings)
        self.write(self.cache/'input_bindings.json', self.bindings)
        self.index = dict(format='v24_complete_recipient_GT_free_native_canonical_v1', complete=True,
            debug=False, recipient_GT_used_in_forward=False, records=[{'id':'DEBUG_record_'+str(i)} for i in range(14102)],
            input_bindings_sha256=self.bindings['content_sha256'],**fields)
        self.write(self.cache/'local/index.json', self.index)
        config=json.loads((Path(__file__).resolve().parents[1]/'config/v25.json').read_text())
        config['arm']='A';config['runtime']['curriculum']=copy.deepcopy(config['arms']['A'])
        self.request = dict(input_cache=str(self.cache.resolve()), config=config)
        self.request['request_sha256'] = _hash(self.request)
        self.write(self.group/'A/request.json', self.request)
        self.full = dict(format='v24_recipient_GT_blind_upper_graphs_v1', active_u=128,
            cases=86, recipient_GT_used_in_forward=False)
        self.common = dict(format=FORMAT, complete=True, debug=False, group_root=str(self.group.resolve()),
            input_cache=str(self.cache.resolve()), request_sha256=self.request['request_sha256'],
            local_index_sha256=hashlib.sha256((self.cache/'local/index.json').read_bytes()).hexdigest(),
            input_bindings_file_sha256=hashlib.sha256((self.cache/'input_bindings.json').read_bytes()).hexdigest(),
            canonical_records=14102, full_upper=self.full,
            context={'contract_sha256':fields['context_contract_sha256']},
            production_optimizer_updates=0, neural_model_created=False)
        self.population=dict(format='v23_all_observed_P_cumulative_native_U_v1',DEBUG=False,
            hidden_subset=False,retained_observations=14102,data_fraction=1.)
        self.population['sha256']=_hash(self.population);self.common['population']=self.population
        self.common['content_sha256'] = _hash(self.common)
        self.write(self.group/'shared_preparation.json', self.common)

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding='utf8')

    def test_common_proof_does_not_require_any_initial_U_prefix(self):
        self.assertFalse((self.group/'A/preparation.json').exists())
        self.assertEqual(validate_shared_preparation(self.group, self.cache), self.common)

    def test_rejects_partial_index_and_changed_full128_evidence(self):
        for change in ('index', 'duplicate_id', 'binding_field', 'full', 'bindings', 'duplicate_case', 'debug'):
            with self.subTest(change=change):
                original_index=(self.cache/'local/index.json').read_bytes()
                original_proof=(self.group/'shared_preparation.json').read_bytes()
                original_bindings=(self.cache/'input_bindings.json').read_bytes()
                try:
                    proof=json.loads(original_proof)
                    if change in ('index','duplicate_id','binding_field'):
                        index=json.loads(original_index)
                        if change=='index':index['records'].pop()
                        elif change=='duplicate_id':index['records'][-1]=index['records'][0]
                        else:index['source_sha256']='f'*64
                        self.write(self.cache/'local/index.json', index)
                        proof['local_index_sha256']=hashlib.sha256((self.cache/'local/index.json').read_bytes()).hexdigest()
                    elif change == 'full':proof['full_upper']['cases']=85
                    elif change in ('bindings','duplicate_case'):
                        bindings=json.loads(original_bindings)
                        if change=='bindings':bindings['case_ids'].pop()
                        else:bindings['case_ids'][-1]=bindings['case_ids'][0]
                        bindings['content_sha256']=_hash({k:v for k,v in bindings.items() if k!='content_sha256'})
                        self.write(self.cache/'input_bindings.json',bindings)
                        proof['input_bindings_file_sha256']=hashlib.sha256((self.cache/'input_bindings.json').read_bytes()).hexdigest()
                    else:proof['debug']=True
                    proof['content_sha256']=_hash({k:v for k,v in proof.items() if k!='content_sha256'})
                    self.write(self.group/'shared_preparation.json',proof)
                    with self.assertRaises(ValueError):validate_shared_preparation(self.group,self.cache)
                finally:
                    (self.cache/'local/index.json').write_bytes(original_index)
                    (self.group/'shared_preparation.json').write_bytes(original_proof)
                    (self.cache/'input_bindings.json').write_bytes(original_bindings)

    def test_only_exact_common_cache_lock_is_replaced_by_per_arm_lock(self):
        import hiercp_v1x.u_bridge_experiment as experiment
        previous=experiment.lock;self.addCleanup(setattr,experiment,'lock',previous)
        paths=[]
        @contextmanager
        def original(path):
            paths.append(Path(path).resolve())
            yield
        for arm in 'ABCD':
            args=SimpleNamespace(mode='prepare',arm=arm,input_cache=self.cache,output=self.group/arm)
            receipt=install_prefix_lock(args,original_lock=original)
            self.assertTrue(receipt['all_other_locks_unchanged'])
            with experiment.lock(self.cache/'.prepare.lock'):pass
            with experiment.lock(self.group/arm/'.v25.lock'):pass
        self.assertEqual(paths[::2],[self.group/arm/'.prepare-prefix.lock' for arm in 'ABCD'])
        self.assertEqual(paths[1::2],[self.group/arm/'.v25.lock' for arm in 'ABCD'])

    def test_prefix_lock_revalidates_immutable_common_proof_before_acquisition(self):
        import hiercp_v1x.u_bridge_experiment as experiment
        previous=experiment.lock;self.addCleanup(setattr,experiment,'lock',previous)
        @contextmanager
        def original(path):yield
        args=SimpleNamespace(mode='prepare',arm='A',input_cache=self.cache,output=self.group/'A')
        install_prefix_lock(args,original_lock=original)
        proof=dict(self.common, debug=True)
        proof['content_sha256']=_hash({k:v for k,v in proof.items() if k!='content_sha256'})
        self.write(self.group/'shared_preparation.json',proof)
        with self.assertRaises(ValueError):
            with experiment.lock(self.cache/'.prepare.lock'):pass

    def test_arm_admission_requires_its_own_completed_initial_U(self):
        for arm,initial in [('A',8),('B',16),('C',32),('D',128)]:
            request=copy.deepcopy(self.request);request['config']['arm']=arm
            request['config']['runtime']['curriculum']=copy.deepcopy(request['config']['arms'][arm])
            request['request_sha256']=_hash({k:v for k,v in request.items() if k!='request_sha256'})
            self.write(self.group/arm/'request.json',request)
            prepared=dict(request_sha256=request['request_sha256'], local_index_sha256=self.common['local_index_sha256'],
                full_upper=self.full,initial_upper=dict(self.full,active_u=initial),context=self.common['context'],
                population=self.population,curriculum=validate_policy(request['config']['runtime']['curriculum']),
                production_optimizer_updates=0,neural_model_created=False)
            self.write(self.group/arm/'preparation.json',prepared)
            self.assertEqual(validate_arm_preparation(self.group,self.cache,arm),prepared)
            prepared['initial_upper']['cases']=85
            self.write(self.group/arm/'preparation.json',prepared)
            with self.assertRaises(ValueError):validate_arm_preparation(self.group,self.cache,arm)

    def test_GT_and_arm_policy_or_population_changes_are_rejected(self):
        arm='B';request=copy.deepcopy(self.request);request['config']['arm']=arm
        request['config']['runtime']['curriculum']=copy.deepcopy(request['config']['arms'][arm])
        request['request_sha256']=_hash({k:v for k,v in request.items() if k!='request_sha256'})
        prepared=dict(request_sha256=request['request_sha256'],local_index_sha256=self.common['local_index_sha256'],
            full_upper=self.full,initial_upper=dict(self.full,active_u=16),context=self.common['context'],
            population=self.population,curriculum=validate_policy(request['config']['runtime']['curriculum']),
            production_optimizer_updates=0,neural_model_created=False)
        for change in ('initial_GT','full_GT','policy','population','config'):
            with self.subTest(change=change):
                selected=copy.deepcopy(request);receipt=copy.deepcopy(prepared)
                if change=='initial_GT':receipt['initial_upper']['recipient_GT_used_in_forward']=True
                elif change=='full_GT':receipt['full_upper']['recipient_GT_used_in_forward']=True
                elif change=='policy':receipt['curriculum']['policy']['initial_u']=8
                elif change=='population':receipt['population']['retained_observations']=14000
                else:
                    selected['config']['epochs']=1
                    selected['request_sha256']=_hash({k:v for k,v in selected.items() if k!='request_sha256'})
                    receipt['request_sha256']=selected['request_sha256']
                self.write(self.group/arm/'request.json',selected)
                self.write(self.group/arm/'preparation.json',receipt)
                with self.assertRaises(ValueError):validate_arm_preparation(self.group,self.cache,arm)

    def test_actual_runtime_profile_is_preserved_for_success_and_failure(self):
        calls=[]
        def publish(path,value):
            calls.append((path,value));self.write(path,value)
        frozen=SimpleNamespace(publish=publish)
        args=SimpleNamespace(output=self.group/'A')
        inputs=SimpleNamespace(preparation_runtime_receipt=lambda:{'actual_RSS_bytes':12345,'checks':4})
        publish_runtime_receipt(frozen,args,inputs,None)
        publish_runtime_receipt(frozen,args,inputs,MemoryError('DEBUG observed pressure'))
        self.assertNotEqual(calls[0][0],calls[1][0])
        self.assertEqual(calls[0][1]['profile']['actual_RSS_bytes'],12345)
        self.assertIsNone(calls[0][1]['failure'])
        self.assertEqual(calls[1][1]['failure'],'MemoryError: DEBUG observed pressure')


if __name__ == '__main__':
    unittest.main()
