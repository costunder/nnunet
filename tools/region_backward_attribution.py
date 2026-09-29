"""Diagnostic autograd-node attribution; no model equations or gradients changed.

Inner forward ranges claim newly created autograd nodes first. Parent ranges
claim only unassigned nodes. CUDA-event durations include launch gaps and hook
overhead; they are not pure kernel durations or uninstrumented production time.
"""
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import torch


def tensor_leaves(value):
    if torch.is_tensor(value):yield value
    elif isinstance(value,dict):
        for item in value.values():yield from tensor_leaves(item)
    elif isinstance(value,(list,tuple)):
        for item in value:yield from tensor_leaves(item)


class Attribution:
    def __init__(self):
        self.labels={};self.events=[];self.handles=[];self.counts=defaultdict(int)

    def tag(self,value,label):
        pending=[t.grad_fn for t in tensor_leaves(value) if t.grad_fn is not None]
        while pending:
            node=pending.pop()
            if node in self.labels:continue
            # Parameter accumulation is shared by branches. Attribute it separately.
            self.labels[node]='parameter_accumulation' if type(node).__name__=='AccumulateGrad' else label
            pending.extend(n for n,_ in node.next_functions if n is not None)
        return value

    @contextmanager
    def ranges(self,net):
        from l0_regions import encoder
        with ExitStack() as stack:
            def wrap(obj,name,label):
                original=getattr(obj,name)
                def call(*args,**kwargs):return self.tag(original(*args,**kwargs),label)
                stack.enter_context(patch.object(obj,name,call))
            wrap(net.local.dense_encoder,'forward','CNN')
            wrap(encoder,'sample_nodes','fine_feature_sampling')
            wrap(encoder,'mass_mean','region_pooling')
            wrap(net.local.core,'_run_local_block','SAGE_blocks')
            wrap(net.local,'forward','L0_readout_and_other')
            wrap(net,'prepare_support','support_L1_L2')
            wrap(net,'predict_embeddings','query_L1_L2')
            yield

    def attach(self,loss):
        self.tag(loss,'loss_and_other')
        for node,label in self.labels.items():
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            state={'called':False}
            def pre(grads,start=start,state=state):
                if state['called']:raise RuntimeError('Autograd node executed twice in one diagnostic')
                state['called']=True;start.record()
            def post(outputs,inputs,end=end):end.record()
            self.handles.extend([node.register_prehook(pre),node.register_hook(post)])
            self.events.append((label,node.name(),start,end,state))

    def report(self):
        torch.cuda.synchronize();by_label=defaultdict(float);by_op=defaultdict(float);counts=defaultdict(int)
        for label,name,start,end,state in self.events:
            if not state['called']:continue
            seconds=start.elapsed_time(end)/1000
            by_label[label]+=seconds;by_op[(label,name)]+=seconds;counts[label]+=1
        total=sum(by_label.values())
        return dict(groups=[dict(group=k,seconds=v,share_of_attributed_percent=100*v/total if total else 0,nodes=counts[k])
                            for k,v in sorted(by_label.items(),key=lambda kv:kv[1],reverse=True)],
                    top_ops=[dict(group=k[0],operation=k[1],seconds=v) for k,v in sorted(by_op.items(),key=lambda kv:kv[1],reverse=True)[:20]],
                    attributed_seconds=total,registered_nodes=len(self.labels),executed_nodes=sum(counts.values()),
                    scope='Autograd-node CUDA-event intervals including diagnostic hook/launch gaps; compare with uninstrumented baseline; not pure kernel time')

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear();self.events.clear();self.labels.clear()
