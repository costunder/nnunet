"""Explicit one-epoch/eight-train/two-validation actual CT lifecycle DEBUG."""
from pathlib import Path
import json,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    import hiercp.model as implementation
    from hiercp_v222 import v1_execution as ex
    from hiercp_v222.v1_cache import configuration
    from tools.v222_runtime_cache import CachedPairDataset
    from tools.v222_runtime_execution import installed as runtime_installed
    from tools.v222_process_loader import installed as process_installed
    from tools.v222_support_snapshot import installed as snapshot_installed,AsyncSaver
    from tools.v222_review_contracts import installed as coordinates_installed
    from tools.v22_rank_objective import OBJECTIVE,configuration as rank_configuration
    from tools.v22_ranking_training import train
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json'
    cfg,base=configuration();debug_cfg=dict(cfg,gnn_epochs=1)
    implementation.EDGE_ATTENTION_WORKSPACE_BYTES=256*1024**2
    torch.cuda.set_per_process_memory_fraction(9e9/torch.cuda.get_device_properties(0).total_memory)
    class DebugDataset(CachedPairDataset):
        def __init__(self,index,partition):
            super().__init__(index,partition);grouped={}
            for row in self.rows:grouped.setdefault(row['case_id'],[]).append(row)
            selected=[];needed=4 if partition=='inner_train' else 1
            for case,rows in sorted(grouped.items()):
                if {r['target'] for r in rows}=={0,1}:
                    selected.extend(next(r for r in rows if r['target']==label) for label in (0,1))
                if len(selected)==needed*2:break
            if len(selected)!=needed*2:raise ValueError('Missing required actual CT DEBUG groups')
            self.rows=selected;self.meta=dict(self.meta,config=debug_cfg)
    class PauseAfterUpdate(AsyncSaver):
        def save(self,state):
            result=super().save(state)
            if state['phase']=='optimization' and state['step']==1:
                (self.root/'STOP_AFTER_BATCH').touch(exist_ok=True)
            return result
    policy=dict(debug=True,training_objective=OBJECTIVE,feature_coordinates='stride4')
    def run(output,resume=None,pause=False):
        with runtime_installed(dict(policy)),process_installed(),snapshot_installed(),coordinates_installed('stride4'), \
             patch.object(ex,'configuration',return_value=(debug_cfg,base)),patch.object(ex,'PairDataset',DebugDataset):
            if pause:
                with patch.object(ex,'Saver',PauseAfterUpdate):return train(cache,output,debug=True,release_unused=True)
            return train(cache,output,resume=resume,debug=True,release_unused=True)
    checkpoint=run(root/'paused',pause=True)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['state']['step']==1 and saved['debug'] and saved['ranking_contract']==rank_configuration()
    completed=run(root/'resumed',resume=checkpoint)
    final=torch.load(completed,map_location='cpu',weights_only=False)
    metrics=json.loads((root/'resumed/epoch_001.json').read_text())
    assert final['debug'] and final['optimization_steps']==4 and final['completed_epochs']==1
    assert len(final['memory']['record_ids'])==8
    assert metrics['all_train_queries_visited']==8 and metrics['validation']['samples']==2
    assert (root/'resumed/ranking_epoch_001.json').is_file()
    result=dict(debug=True,real_CT=True,full_training=False,train_records=8,validation_records=2,
                physical_batch=final['physical_batch'],epochs=1,paused_after_step=1,completed_steps=4,
                all_debug_queries_visited_once=True,refresh_validation_best_checkpoint_final_memory_completed=True,
                final_checkpoint_feature_coordinates=final['feature_coordinates'],training_objective=final['training_objective'],
                validation=metrics['validation'])
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
