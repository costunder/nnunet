import copy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
from l0_regions.execution_pipeline import CheckpointPipeline,DeviceBatchCache,packed_cpu_snapshot,gradient_check_batched,verify_pipeline_upgrade
from l0_regions.training import hash_state,FORMAT


class PipelineChecks(unittest.TestCase):
    def test_device_cache_lru_oversize_and_zero_budget(self):
        from types import SimpleNamespace
        from l0_regions.resident import signature
        class Batch:
            def __init__(self,n):
                self.x=torch.arange(n,dtype=torch.float32)
                self._verified_signature=signature(self)
            def to(self,device):return Batch(len(self.x))
        budget=SimpleNamespace(cuda_bytes=1000,check=lambda:None)
        a,b,c=Batch(4),Batch(4),Batch(4)
        with patch('torch.cuda.memory_allocated',return_value=0):
            cache=DeviceBatchCache(32,budget)
            first=cache.get(a);cache.get(b)
            self.assertIs(cache.get(a),first)
            cache.get(c)
            self.assertEqual(set(cache.items),{id(a),id(c)})
            self.assertEqual(cache.bytes,32)
            whole=cache.get(Batch(100))
            self.assertEqual(len(whole.x),100)
            self.assertEqual(cache.bytes,0)
            disabled=DeviceBatchCache(0,budget)
            self.assertEqual(len(disabled.get(a).x),4)
            self.assertEqual(disabled.bytes,0)
            self.assertFalse(disabled.items)

    def test_channels_last_snapshot_format(self):
        value=torch.randn(2,3,4,5,6).to(memory_format=torch.channels_last_3d)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            source=value.to(device);snap=packed_cpu_snapshot(source)
            self.assertEqual(source.stride(),snap.stride())
            self.assertTrue(torch.equal(source.cpu(),snap))

    def test_workspace_forward_backward_exact_and_restored(self):
        from l0_regions.sparse import workspace,stable_spmm,_WORKSPACE
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            e=torch.stack([torch.arange(1024)%64,torch.arange(1024)%53]).to(device)
            a=torch.sparse_coo_tensor(e,torch.ones(1024,device=device),(64,64)).coalesce().to_sparse_csr()
            at=a.transpose(0,1).to_sparse_csr();x=torch.randn(64,128,device=device,requires_grad=True)
            results=[]
            for size in (4096,2**20):
                with workspace(size):out=stable_spmm(a,at,x)
                results.append((out.detach().clone(),torch.autograd.grad(out.square().sum(),x)[0]))
            for a,b in zip(*results):self.assertTrue(torch.equal(a,b))
            self.assertEqual(_WORKSPACE.get(),64*2**20)

    def test_snapshot_owns_cpu_storage(self):
        a=torch.arange(12.).reshape(3,4);v=dict(a=a,b=[a[:,1:],dict(n=2)])
        snap=packed_cpu_snapshot(v);expected=hash_state(snap)
        a.add_(5);v['b'][1]['n']=8
        self.assertEqual(hash_state(snap),expected)
        self.assertFalse(torch.equal(a,snap['a']))

    def test_writer_is_immutable_ordered_and_content_verified(self):
        net=torch.nn.Linear(3,2);opt=torch.optim.AdamW(net.parameters())
        state=dict(step=1,phase='optimization',memory={'x':torch.ones(2)},plan=None,best=None,memory_parts=[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with CheckpointPipeline(root,'overlapped',None,hash_state,FORMAT) as writer:
                writer.save(net,opt,state,{'test':True})
                state['step']=2;state['memory']['x'].add_(1)
                with torch.no_grad():net.weight.add_(1)
                writer.flush()
                first=torch.load(root/'checkpoint_latest.pt',weights_only=False)
                self.assertEqual(first['state']['step'],1)
                self.assertTrue(torch.equal(first['state']['memory']['x'],torch.ones(2)))
                writer.save(net,opt,state,{'test':True},wait=True)
            last=torch.load(root/'checkpoint_latest.pt',weights_only=False)
            digest=last.pop('content_sha256')
            self.assertEqual(hash_state(last),digest)
            self.assertEqual(last['state']['step'],2)

    def test_static_snapshot_reuse_and_version_invalidation(self):
        net=torch.nn.Linear(1,1);opt=torch.optim.AdamW(net.parameters())
        state=dict(step=0,phase='refresh_memory',memory=torch.ones(2),plan=None,best=None,memory_parts=[torch.ones(1)])
        with tempfile.TemporaryDirectory() as tmp:
            with CheckpointPipeline(Path(tmp),'overlapped',None,hash_state,FORMAT) as writer:
                writer.save(net,opt,state,{},wait=True);first=writer.static['memory'][1]
                part=next(iter(writer.parts.values()))[0]
                state['memory_parts'].append(torch.ones(1)*2)
                writer.save(net,opt,state,{},wait=True)
                self.assertIs(writer.static['memory'][1],first)
                self.assertIs(next(iter(writer.parts.values()))[0],part)
                state['memory'].add_(1);writer.save(net,opt,state,{},wait=True)
                self.assertIsNot(writer.static['memory'][1],first)

    def test_writer_failure_is_propagated(self):
        net=torch.nn.Linear(1,1);opt=torch.optim.AdamW(net.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            def fail(_):raise OSError('disk failed')
            with self.assertRaisesRegex(OSError,'disk failed'):
                with CheckpointPipeline(Path(tmp),'overlapped',None,fail,FORMAT) as writer:
                    writer.save(net,opt,dict(step=1,phase='optimization'),{})

    def test_gradient_checks_and_no_mutation(self):
        net=torch.nn.Linear(2,2);net(torch.ones(1,2)).sum().backward()
        before=[p.grad.clone() for p in net.parameters()]
        gradient_check_batched(net)
        self.assertTrue(all(torch.equal(a,p.grad) for a,p in zip(before,net.parameters())))
        net.weight.grad[0,0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'nonfinite'):gradient_check_batched(net)
        net.weight.grad=None
        with self.assertRaisesRegex(RuntimeError,'missing'):gradient_check_batched(net)

    def test_upgrade_rejects_learning_changes(self):
        old=dict(batch=32,support_training={'patients':16},source={'core':{},'runtime':{'l0_regions/training.py':'old'}})
        new=dict(old,execution_pipeline={'mode':'overlapped'},source={'core':{},'runtime':{'l0_regions/training.py':'new','l0_regions/execution_pipeline.py':'new'}})
        with patch('l0_regions.execution_upgrade.blob_hash',side_effect=lambda revision,path:'unknown' if revision=='4389182' else 'old'):
            self.assertEqual(verify_pipeline_upgrade(old,new),'10287dd')
            for key,value in [('batch',16),('support_training',{'patients':8})]:
                with self.assertRaises(ValueError):verify_pipeline_upgrade(old,dict(new,**{key:value}))

if __name__=='__main__':unittest.main()
