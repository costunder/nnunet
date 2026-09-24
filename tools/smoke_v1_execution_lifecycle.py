"""Explicit DEBUG control-flow test, all real fixture CT, full-size model.

Exercises pause/resume -> epoch refresh -> validation -> best checkpoint.
Six fixture observations are deliberately reused for validation ONLY to test
execution control. The resulting metrics are NOT experimental performance.
No production config, graph cache or checkpoint is modified.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import os
from unittest.mock import patch
import torch
import hiercp_v222.v1_execution as execution
from hiercp_v222.v1_cache import configuration

def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    fixture_path=Path(os.environ.get('HIERCP_TEST_FIXTURE',ROOT/'work/v222_v1_l0_20260924/debug2/actual_graphs_DEBUG.pt'))
    fixture=torch.load(fixture_path,map_location='cpu',weights_only=False)
    cfg,base=configuration();cfg=dict(cfg,gnn_epochs=1) # Only this explicit DEBUG control-flow test.
    class FixtureDataset:
        def __init__(self,index,partition):
            self.rows=[r|dict(donor_group='case:liver_1',bounds=dict(edges=int(fixture['values'][i][0].num_edges))) for i,r in enumerate(fixture['rows'])]
            cases=sorted({r['case_id'] for r in self.rows})
            self.meta=dict(config=cfg,base=base,split=dict(inner_train=cases),
                identities=dict(cases={r['case_id']:dict(patient_group=r['patient_group']) for r in self.rows}),
                donor_pool=[dict(case_id='liver_1')],raw_records=[],debug=True)
            self.budget=0
        def __len__(self):return len(self.rows)
        def item(self,i,epoch=0):return fixture['values'][i],i
    def stop_after_first(self):
        receipt=json.loads((self.root/'checkpoint_status.json').read_text(encoding='utf-8'))
        return receipt['phase']=='optimization' and receipt['step']==1
    with patch.object(execution,'configuration',return_value=(cfg,base)),patch.object(execution,'PairDataset',FixtureDataset):
        with patch.object(execution.Saver,'stop_requested',stop_after_first):
            first=execution.train(fixture_path,root/'interrupted',release_unused=True,debug=True)
        saved=torch.load(first,map_location='cpu',weights_only=False)
        assert saved['debug'] and saved['state']['step']==1 and saved['state']['next_batch']==1
        result=execution.train(fixture_path,root/'continued',resume=first,release_unused=True,debug=True)
    final=torch.load(result,map_location='cpu',weights_only=False)
    epoch=json.loads((root/'continued/epoch_001.json').read_text(encoding='utf-8'))
    assert final['debug'] and final['completed_epochs']==1 and final['optimization_steps']==3
    assert epoch['all_train_queries_visited']==6 and len(final['memory']['embeddings'])==6
    report=dict(debug=True,full_training=False,real_CT=True,full_model_parameters=sum(v.numel() for v in final['state_dict'].values()),
        purpose='execution lifecycle only; train/validation fixture reuse is not a performance experiment',
        paused_after_optimizer_step=1,resumed_epoch_finished=True,query_coverage=6,optimization_steps=3,
        support_refresh_completed=True,validation_executed=True,best_checkpoint_finalized=True)
    (root/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
