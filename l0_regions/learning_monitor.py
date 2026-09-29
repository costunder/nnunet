"""Display-only learning diagnostics; never change RNG, optimizer or checkpoint state."""
from collections import deque
import torch


class LearningMonitor:
    def __init__(self,net):
        self.losses=deque(maxlen=20)
        prefixes={'CNN':'local.core.dense_encoder','SAGE1':'local.core.blocks.0.conv',
                  'SAGE2':'local.core.blocks.1.conv','SAGE3':'local.core.blocks.2.conv',
                  'L1':'l1','L2':'l2'}
        params=list(net.named_parameters());self.probes={}
        for label,prefix in prefixes.items():
            selected=next(((n,p) for n,p in params if n.startswith(prefix+'.') and p.requires_grad and p.ndim>=2),None)
            if selected is None:raise ValueError(f'Missing monitored module {label}')
            name,p=selected
            # Only a labelled probe, not a claim that every parameter changed.
            indices=torch.linspace(0,p.numel()-1,min(32,p.numel()),device=p.device).long()
            self.probes[label]=(name,p,indices)

    @torch.no_grad()
    def before_step(self):
        return {label:p.reshape(-1).index_select(0,idx).clone()
                for label,(_,p,idx) in self.probes.items()}

    @torch.no_grad()
    def after_step(self,loss,terms,grad_norm,before,lr):
        values=[loss.detach(),terms['ranking_loss'].detach(),terms['observation_auxiliary_loss'].detach(),
                terms['alignment_loss'].detach(),grad_norm.detach()]
        for label,(_,p,idx) in self.probes.items():
            values.append((p.reshape(-1).index_select(0,idx)-before[label]).abs().max())
        measured=torch.stack([v.float() for v in values]).cpu().tolist()
        self.losses.append(measured[0]);delta=dict(zip(self.probes,measured[5:]))
        return dict(loss=measured[0],loss_mean20=sum(self.losses)/len(self.losses),
                    loss_window_updates=len(self.losses),ranking_loss=measured[1],
                    observation_auxiliary_loss=measured[2],alignment_loss=measured[3],
                    gradient_norm_before_clip=measured[4],learning_rate=float(lr),
                    optimizer_step_completed=True,probe_max_abs_delta=delta,
                    probe_parameters={k:v[0] for k,v in self.probes.items()},
                    probe_scope='up to 32 deterministic elements of one weight tensor per module; not whole-model coverage',
                    loss_window_scope='last up to 20 updates in this process/epoch segment; reset on resume')

    @staticmethod
    def postfix(health,step):
        changes=health['probe_max_abs_delta']
        return dict(step=step,loss=f"{health['loss']:.3f}",avg20=f"{health['loss_mean20']:.3f}",
                    grad=f"{health['gradient_norm_before_clip']:.2g}",
                    probe=f"{sum(v>0 for v in changes.values())}/{len(changes)}",lr=f"{health['learning_rate']:.1g}")


def validation_line(epoch,metrics,best,improved):
    recalls=' '.join(f"R@{k.rsplit('_',1)[-1]}={v:.3f}" for k,v in metrics.items() if k.startswith('ranking_recall_at_'))
    return (f"Validation epoch {epoch} | rank_loss={metrics['ranking_pairwise_loss']:.4f} "
            f"MRR={metrics['ranking_mrr']:.3f} {recalls} | best={best:.4f}"
            + (' NEW_BEST' if improved else '')
            + ' | observed-tumor ranking; not segmentation Dice')
