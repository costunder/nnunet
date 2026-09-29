"""Synthetic checkpoint rejection tests; real CUDA resume uses the CLI smoke."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from l0_regions.training import FORMAT,hash_state,load_checkpoint

class RegionResumeContract(unittest.TestCase):
    def setUp(self):
        self.folder=Path(tempfile.mkdtemp(prefix='region_resume_contract_',dir=Path(__file__).resolve().parents[1]/'work'))
        self.identity=dict(debug=True,cache_sha256='fixture',epochs=1,source={'runtime':'fixture'})
        self.payload=dict(format=FORMAT,identity=self.identity,model={'weight':torch.tensor([1.])},
            optimizer={'step':torch.tensor(2)},state={'phase':'optimization','step':2},rng={'torch':torch.tensor([42],dtype=torch.uint8)})
        self.payload['content_sha256']=hash_state(self.payload)
    def save(self,payload,name):
        path=self.folder/name
        with path.open('xb') as f:torch.save(payload,f)
        return path
    def test_valid_round_trip(self):
        result=load_checkpoint(self.save(self.payload,'valid.pt'),self.identity)
        self.assertEqual(result['state']['step'],2)
    def test_changed_model_optimizer_cursor_rng_rejected(self):
        for key in ('model','optimizer','state','rng'):
            with self.subTest(key=key):
                bad=copy.deepcopy(self.payload)
                if key=='state':bad[key]['step']+=1
                else:next(iter(bad[key].values())).add_(1)
                with self.assertRaisesRegex(ValueError,'contents changed'):
                    load_checkpoint(self.save(bad,key+'.pt'),self.identity)
    def test_debug_cannot_resume_as_full(self):
        with self.assertRaisesRegex(ValueError,'exact region resume'):
            load_checkpoint(self.save(self.payload,'debug.pt'),dict(self.identity,debug=False))
    def test_other_cache_and_runtime_cannot_resume(self):
        path=self.save(self.payload,'identity.pt')
        for key,value in (('cache_sha256','other'),('source',{'runtime':'changed'}),('epochs',40)):
            with self.assertRaisesRegex(ValueError,'exact region resume'):
                load_checkpoint(path,dict(self.identity,**{key:value}))
    def test_legacy_gat_cannot_resume(self):
        bad=copy.deepcopy(self.payload);bad['format']='v222_v1_exact_execution_resume_v1'
        with self.assertRaisesRegex(ValueError,'exact region resume'):
            load_checkpoint(self.save(bad,'gat.pt'),self.identity)

if __name__=='__main__':unittest.main(verbosity=2)
