"""CPU fixtures for exact-byte loading and full-support handoff contracts."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from l0_regions import resident


class SingleRead(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'record.pt';self.binding={'record':'DEBUG synthetic'}
        self.item={'binding':self.binding,'value':torch.arange(6)}
        torch.save(self.item,self.path)
        self.manifest={'binding':self.binding,'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}
        self.path.with_suffix('.json').write_text(json.dumps(self.manifest))
    def tearDown(self):self.temp.cleanup()
    def test_exact_hashed_bytes_deserialized_once_and_validator_invoked(self):
        calls=[];original=Path.read_bytes
        def read(path):
            if path==self.path:calls.append(path)
            return original(path)
        with patch.object(Path,'read_bytes',read),patch.object(resident,'VerifiedItem',side_effect=lambda item:item) as validate:
            item=resident.load(self.path,self.binding)
        self.assertEqual(len(calls),1);self.assertEqual(validate.call_count,1)
        self.assertTrue(torch.equal(item['value'],self.item['value']))
    def test_corrupt_bytes_fail_before_deserialization(self):
        self.path.write_bytes(b'DEBUG corrupt')
        with patch.object(torch,'load',side_effect=AssertionError('must not deserialize')):
            with self.assertRaisesRegex(ValueError,'Corrupt'):resident.load(self.path,self.binding)
    def test_different_binding_rejected(self):
        with self.assertRaisesRegex(ValueError,'Different record'):resident.load(self.path,{'record':'other'})
    def test_structural_validation_not_bypassed(self):
        with patch.object(resident,'VerifiedItem',side_effect=ValueError('invalid coverage')):
            with self.assertRaisesRegex(ValueError,'coverage'):resident.load(self.path,self.binding)


class EdgeCount(unittest.TestCase):
    def check(self,device):
        from l0_regions.data import unique_edge_count
        generator=torch.Generator().manual_seed(19)
        for size in (0,1,7,1000):
            edge=torch.stack((torch.randint(19,(size,),generator=generator),torch.randint(31,(size,),generator=generator))).to(device)
            edge=torch.cat((edge,edge[:,:min(size,4)]),1)
            self.assertEqual(unique_edge_count(edge,19,31),len(torch.unique(edge.T,dim=0)))
        # A packed key without the overflow guard would wrap and collide.
        edge=torch.tensor([[0,2**62,0],[0,0,0]],device=device)
        self.assertEqual(unique_edge_count(edge,2**62+1,4),2)
        for edge in (torch.tensor([[-1],[0]],device=device),torch.tensor([[19],[0]],device=device)):
            with self.assertRaisesRegex(ValueError,'endpoint'):unique_edge_count(edge,19,31)
    def test_cpu(self):self.check('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda(self):self.check('cuda')


if __name__=='__main__':unittest.main()
