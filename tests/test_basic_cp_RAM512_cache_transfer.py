"""CPU DEBUG fixtures; never connects, starts or signals remote processes."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location('debug_'+name,ROOT/'tools'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def requests():
    arms=['multi5','blend50','multi10','candidates512','candidates1024','candidates2048','multi7']
    return {str(i):dict(root='/old/GPU'+str(i)+'_'+arm,arm=arm,gpu=i,gpu_uuid='DEBUG_UUID_'+str(i),
        cpu_affinity=list(range(8*i,8*i+8)),GPU_memory_total_MiB=49140,baseline={'epochs':250,'batch':2},
        split={'train':list(range(105)),'val':list(range(26))},bank_sha256='a'*64,online_seed=42,
        code_sha256={'old.py':'a'*64},commit='a'*40)for i,arm in enumerate(arms)}

def transfer(old,swaps):
    mapping={g:g for g in old}
    for a,b in swaps:mapping[a],mapping[b]=mapping[b],mapping[a]
    result={}
    for gpu,prior in mapping.items():
        q=copy.deepcopy(old[prior]);q.update(root='/new/GPU'+gpu+'_'+q['arm'],gpu=int(gpu),
            gpu_uuid=old[gpu]['gpu_uuid'],cpu_affinity=old[gpu]['cpu_affinity'],GPU_memory_total_MiB=old[gpu]['GPU_memory_total_MiB'],
            code_sha256={'new.py':'b'*64},commit='b'*40,GPU_placement_receipt={'path':'/new/placement.json','sha256':'c'*64})
        result[gpu]=q
    return result,mapping

class CacheTransferTests(unittest.TestCase):
    def setUp(self):self.adopter=load('adopt_basic_cp_RAM512_cache_transfer');self.old=requests()

    def test_four_full_arm_bijections(self):
        for swaps in ([],[('0','5')],[('4','6')],[('0','5'),('4','6')]):
            with self.subTest(swaps=swaps):
                new,expected=transfer(self.old,swaps);actual=self.adopter.arm_mapping(self.old,new)
                self.assertEqual(actual,expected);self.adopter.validate_request_transfer(self.old,new,actual)
                renames=self.adopter.cache_renames(self.old,new,actual)
                self.assertEqual(len(renames),2*len(swaps));self.assertEqual(len(set(renames.values())),len(renames))

    def test_deferred_receipt_is_operational(self):
        new,mapping=transfer(self.old,[]);new['0']['GPU_queue_deferred_transfer']={'path':'/new/queue.json','sha256':'b'*64}
        self.adopter.validate_request_transfer(self.old,new,mapping)

    def lifetime_fixture(self):
        from hiercp_v1x import basic_cp_ablation_resources as native
        new,mapping=transfer(self.old,[])
        before={'path':'/own/old_lifetime.json','sha256':'d'*64,'expires_at_unix':1000.0}
        after={'path':'/own/new_lifetime.json','sha256':'e'*64,'expires_at_unix':10000000000.0}
        for q in self.old.values():q['host_lifetime']=copy.deepcopy(before)
        for q in new.values():q['host_lifetime']=copy.deepcopy(after)
        def witness(ref,job):
            return dict(format='v25_owned_GPU_allocation_v1',owner='aicompetition06',UID=41833,
                node='ece-a6gpu6',job_id=job,job_state='R',expires_at_unix=ref['expires_at_unix'])
        values={str(Path(before['path'])):witness(before,'129864.ECE-util1'),str(Path(after['path'])):witness(after,'130001.ECE-util1')}
        actual=json.dumps({'DEBUG_actual_PBS_read':True})
        calls=[]
        def validate(ref,value,raw):
            calls.append((ref,value,raw))
            if raw!=actual:raise RuntimeError('DEBUG changed actual PBS proof')
            return {'job_id':value['job_id'],'state':'R','DEBUG_validator_fixture':True}
        helper=SimpleNamespace(validate_host_lifetime=validate)
        return new,mapping,before,after,values,actual,calls,helper,native

    def test_changed_lifetime_requires_independent_new_running_PBS_validation(self):
        new,mapping,before,after,values,actual,calls,helper,native=self.lifetime_fixture()
        with (patch.object(self.adopter,'pin',side_effect=lambda path,sha:Path(path)),
              patch.object(self.adopter,'read',side_effect=lambda path:values[str(path)]),
              patch.object(native,'_pbs_query',return_value=actual)as query):
            proof=self.adopter.validate_request_transfer(self.old,new,mapping,helper=helper)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],after)
        self.assertEqual(calls[0][2],actual)
        self.assertTrue(proof['new_running_job_independently_queried'])
        self.assertEqual(proof['old_reference'],before)
        query.assert_called_once_with(['/opt/pbs/bin/qstat','-f','-F','json','130001.ECE-util1'],
            timeout=native.PBS_QUERY_TIMEOUT_SECONDS)

    def test_unverified_lifetime_change_wrong_owner_node_state_or_expiry_is_rejected(self):
        mutations=[('owner','other'),('UID',0),('node','different-host'),('job_state','Q'),
                   ('job_id','invalid-job'),('expires_at_unix',1000.0)]
        for field,value in mutations:
            with self.subTest(field=field):
                new,mapping,before,after,values,actual,calls,helper,native=self.lifetime_fixture()
                values[str(Path(after['path']))][field]=value
                with (patch.object(self.adopter,'pin',side_effect=lambda path,sha:Path(path)),
                      patch.object(self.adopter,'read',side_effect=lambda path:values[str(path)]),
                      patch.object(native,'_pbs_query',return_value=actual)as query,
                      self.assertRaises(ValueError)):
                    self.adopter.validate_request_transfer(self.old,new,mapping,helper=helper)
                query.assert_not_called()

    def test_mixed_cohort_lifetimes_and_failed_actual_PBS_observation_are_fatal(self):
        new,mapping,before,after,values,actual,calls,helper,native=self.lifetime_fixture()
        new['3']['host_lifetime']=before
        with self.assertRaisesRegex(ValueError,'one exact host lifetime'):
            self.adopter.validate_request_transfer(self.old,new,mapping,helper=helper)
        new['3']['host_lifetime']=after
        with (patch.object(self.adopter,'pin',side_effect=lambda path,sha:Path(path)),
              patch.object(self.adopter,'read',side_effect=lambda path:values[str(path)]),
              patch.object(native,'_pbs_query',side_effect=TimeoutError('DEBUG PBS unavailable')),
              self.assertRaises(TimeoutError)):
            self.adopter.validate_request_transfer(self.old,new,mapping,helper=helper)

    def test_actual_lifetime_validator_rejection_cannot_be_bypassed(self):
        new,mapping,before,after,values,actual,calls,helper,native=self.lifetime_fixture()
        with (patch.object(self.adopter,'pin',side_effect=lambda path,sha:Path(path)),
              patch.object(self.adopter,'read',side_effect=lambda path:values[str(path)]),
              patch.object(native,'_pbs_query',return_value='DEBUG_nonrunning_PBS'),
              self.assertRaisesRegex(RuntimeError,'actual PBS proof')):
            self.adopter.validate_request_transfer(self.old,new,mapping,helper=helper)

    def test_unchanged_host_reference_keeps_legacy_transfer_without_extra_query(self):
        from hiercp_v1x import basic_cp_ablation_resources as native
        new,mapping=transfer(self.old,[])
        with patch.object(native,'_pbs_query',side_effect=AssertionError('unnecessary PBS read')):
            self.assertIsNone(self.adopter.validate_request_transfer(self.old,new,mapping))

    def test_seven_scientific_or_physical_mutations_rejected(self):
        for field,bad in [('gpu_uuid','another GPU'),('cpu_affinity',[0]),('GPU_memory_total_MiB',1),
            ('baseline',{'epochs':249,'batch':2}),('bank_sha256','b'*64),('online_seed',43),('unknown_science_option',True)]:
            with self.subTest(field=field):
                new,mapping=transfer(self.old,[('0','5')]);new['5'][field]=bad
                with self.assertRaises(ValueError):self.adopter.validate_request_transfer(self.old,new,mapping)

    def test_four_invalid_populations_rejected(self):
        for mutation in ('duplicate_arm','missing_slot','wrong_gpu','unknown_arm'):
            with self.subTest(mutation=mutation):
                new,_=transfer(self.old,[])
                if mutation=='duplicate_arm':new['5']['arm']=new['0']['arm']
                elif mutation=='missing_slot':new.pop('1')
                elif mutation=='wrong_gpu':new['5']['gpu']=4
                else:new['5']['arm']='tiny_debug_arm'
                with self.assertRaises(ValueError):self.adopter.arm_mapping(self.old,new)

    def test_unknown_latest_provenance_is_preserved(self):
        metadata={'saved_resume_epoch':1,'latest_on_tmpfs_N':None,'epochs_may_repeat_after_tmpfs_loss':None};before=copy.deepcopy(metadata)
        self.assertTrue(self.adopter.whole_unfinished_best_metadata(metadata,{'checkpoint_completed_epoch':1}));self.assertEqual(metadata,before)

    def test_nine_latest_replay_combinations(self):
        for latest,replay,valid in [(7,2,True),(5,0,True),(None,None,True),(None,0,False),(4,0,False),
            (251,246,False),(True,0,False),(7,None,False),(7,1,False)]:
            with self.subTest(latest=latest,replay=replay):
                metadata={'saved_resume_epoch':5,'latest_on_tmpfs_N':latest,'epochs_may_repeat_after_tmpfs_loss':replay}
                if valid:self.assertTrue(self.adopter.whole_unfinished_best_metadata(metadata,{'checkpoint_completed_epoch':5}))
                else:
                    with self.assertRaises(ValueError):self.adopter.whole_unfinished_best_metadata(metadata,{'checkpoint_completed_epoch':5})

    def test_five_fabricated_saved_epochs_rejected(self):
        for saved in (None,0,True,6,251):
            with self.subTest(saved=saved):
                with self.assertRaises(ValueError):self.adopter.whole_unfinished_best_metadata(
                    {'saved_resume_epoch':saved,'latest_on_tmpfs_N':None,'epochs_may_repeat_after_tmpfs_loss':None},
                    {'checkpoint_completed_epoch':5})

    def temporary(self):
        owned=ROOT/'outputs';owned.mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix='DEBUG_cache_transfer_',dir=owned)
        self.assertTrue(Path(temporary.name).resolve().is_relative_to(owned.resolve()))
        return temporary

    def test_directory_transfer_preserves_ten_files_and_two_inodes(self):
        with self.temporary()as directory:
            root=Path(directory);renames={'GPU0_multi5':'GPU5_multi5','GPU5_candidates2048':'GPU0_candidates2048'};before={};inodes={}
            for name in renames:
                folder=root/name;folder.mkdir();inodes[name]=folder.stat().st_ino
                for filename in ('CURRENT.pth','CURRENT.json','PENDING.json','entry.npz','runtime/native.py'):
                    path=folder/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('DEBUG:'+name+filename).encode())
                    before[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(self.adopter,'CACHE',root),patch.object(self.adopter,'exact',lambda p,directory=False:Path(p)),patch.object(self.adopter,'sync',lambda p:None):
                self.adopter.relocate_cache_folders(renames)
            for relative,checksum in before.items():
                target=root/self.adopter.relocated_relative(relative,renames);self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),checksum)
            for source,target in renames.items():self.assertFalse((root/source).exists());self.assertEqual((root/target).stat().st_ino,inodes[source])

    def test_existing_target_prevents_every_rename(self):
        with self.temporary()as directory:
            root=Path(directory);renames={'GPU0_multi5':'GPU5_multi5','GPU5_candidates2048':'GPU0_candidates2048'}
            for name in (*renames,'GPU0_candidates2048'):(root/name).mkdir()
            with patch.object(self.adopter,'CACHE',root),patch.object(self.adopter,'exact',lambda p,directory=False:Path(p)),patch.object(self.adopter,'sync',lambda p:None):
                with self.assertRaises(ValueError):self.adopter.relocate_cache_folders(renames)
            self.assertTrue(all((root/name).exists()for name in renames));self.assertFalse((root/'GPU5_multi5').exists())

    def test_history_not_retargeted_and_traversal_rejected(self):
        renames={'GPU0_multi5':'GPU5_multi5'};original='history/previous/GPU0_multi5/CURRENT.json'
        self.assertEqual(self.adopter.relocated_relative(original,renames).replace('\\','/'),original)
        with self.assertRaises(ValueError):self.adopter.relocated_relative('../CURRENT.pth',renames)

    def test_old_writer_dynamic_manifest_is_exact(self):
        code=Path('/own/current/code');manifest=Path('/own/current/launch.json')
        command=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',str(code/self.adopter.HELPER),
            '--manifest',str(manifest),'--output',str(self.adopter.CACHE),'--unfinished-checkpoint','best','--resume-cache']
        self.adopter.old_helper_command(command,code,manifest)
        with self.assertRaises(ValueError):self.adopter.old_helper_command(command,code,Path('/different/launch.json'))

    def test_reused_pid_never_admits_replacement(self):
        class Reused:
            def create_time(self):return 2.
            def uids(self):raise AssertionError('Replacement UID must never be read/admitted')
        with patch.object(self.adopter.psutil,'Process',lambda pid:Reused()):
            self.assertTrue(self.adopter.released({'UID':41833,'pid':123,'birth':1.,'command':['DEBUG']})['replacement_not_admitted'])

    def test_same_UI2_wrapper_only_retargets_manifest(self):
        keeper=load('hold_basic_cp_RAM512_transfer');group=Path('/dev/shm/aicompetition06_basic_cp_ablation_20261013/Basic_CP_DEBUG_future_123')
        source=keeper.viewer_wrapper(group,group/'code');compile(source,'DEBUG_UI2','exec')
        for required in ('watch_basic_cp_RAM512.py','--manifest','--code-root',repr(str(group))):self.assertIn(required,source)
        self.assertNotIn('print(',source);self.assertNotIn('format(',source)

if __name__=='__main__':unittest.main()
