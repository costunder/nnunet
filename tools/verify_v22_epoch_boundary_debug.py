"""DEBUG lifecycle fixtures on a caller's actual CUDA-updated full native model.

Epoch numbers and metrics below are control inputs, never training results.
The fresh output folder is owned by this test. No OS/process signals are sent.
"""
from pathlib import Path
import shutil
import time
from unittest.mock import patch


def verify(owner,root):
    import torch
    from nnunetv2.training.logging.nnunet_logger import MetaLogger
    from tools.v22_online_checkpoint import file_hash,validate_checkpoint
    root=Path(root);root.mkdir(exist_ok=False)
    if next(owner.network.parameters()).device.type!='cuda':raise AssertionError('Actual CUDA model required')
    if not owner.optimizer.state:raise AssertionError('Actual SGD update must precede lifecycle tests')
    owner.output_folder=str(root);owner.current_epoch=0;owner._best_ema=None
    if hasattr(owner,'_online_best'):del owner._online_best
    latest=root/'checkpoint_latest.pth';best=root/'checkpoint_best.pth'
    rows=[]

    def prepare(epoch,metric):
        # Fill control log history so the installed logger's index and plotting
        # contracts are exercised at 125/250 without pretending to train epochs.
        owner.current_epoch=epoch;owner.logger=MetaLogger(str(root),False)
        for i in range(epoch+1):
            for key,value in [('epoch_start_timestamps',time.time()),('epoch_end_timestamps',time.time()),
                              ('train_losses',1.),('val_losses',1.),('dice_per_class_or_region',[metric,metric]),
                              ('ema_fg_dice',metric),('lrs',owner.optimizer.param_groups[0]['lr'])]:
                owner.logger.log(key,value,i)
            owner.logger.local_logger.log('mean_fg_dice',metric,i)

    # Fresh first epoch cannot legitimately have an earlier best checkpoint.
    for epoch,metric,expected_best in [(0,.8,1),(124,.7,1),(124,.9,125),(249,.8,125),(249,.95,250)]:
        old=file_hash(best) if best.exists() else None
        prepare(epoch,metric);owner.on_epoch_end()
        value=torch.load(latest,map_location='cpu',weights_only=False)
        validate_checkpoint(value,owner._online_contract())
        if value['current_epoch']!=epoch+1 or value['online_best']['epoch']!=expected_best:
            raise AssertionError('Completed cursor/best mismatch')
        if expected_best!=epoch+1 and file_hash(best)!=old:raise AssertionError('Historical best changed')
        owner.load_checkpoint(latest)
        if owner.current_epoch!=epoch+1:raise AssertionError('Loaded cursor mismatch')
        rows.append(dict(completed_epoch=epoch+1,controlled_metric=metric,best_epoch=expected_best,
                         actual_file_load=True,latest_sha256=file_hash(latest),best_sha256=file_hash(best)))
        del value

    baseline={p.name:file_hash(p) for p in (latest,best)}
    prepare(249,.97)
    # A real parent error after save requests must not publish a completed epoch.
    with patch.object(owner.logger,'plot_progress_png',side_effect=RuntimeError('DEBUG logger failure')):
        try:owner.on_epoch_end()
        except RuntimeError as error:
            if str(error)!='DEBUG logger failure':raise
        else:raise AssertionError('Logger failure did not propagate')
    if {p.name:file_hash(p) for p in (latest,best)}!=baseline:raise AssertionError('Failed epoch was published')
    owner.load_checkpoint(latest)

    prepare(249,.98)
    usage=shutil.disk_usage(root)
    # Only telemetry is injected; production reserve, writer and payload remain intact.
    with patch('tools.v22_online_storage.shutil.disk_usage',return_value=type(usage)(usage.total,usage.total,0)):
        try:owner.on_epoch_end()
        except OSError as error:
            if 'disk admission failed' not in str(error):raise
        else:raise AssertionError('Storage failure did not propagate')
    if {p.name:file_hash(p) for p in (latest,best)}!=baseline:raise AssertionError('Storage failure changed checkpoint')
    owner.load_checkpoint(latest)
    if list(root.glob('.online-checkpoint-*')):raise AssertionError('Orphan write temporary')

    # Installed parent on_train_end saves final and removes only this test's latest.
    owner.dataloader_train=None;owner.dataloader_val=None
    owner.on_train_end()
    final=root/'checkpoint_final.pth'
    value=torch.load(final,map_location='cpu',weights_only=False)
    validate_checkpoint(value,owner._online_contract())
    if value['current_epoch']!=250 or owner.current_epoch!=250:raise AssertionError('Final cursor mismatch')
    owner.load_checkpoint(final)
    return dict(debug=True,actual_CUDA_updated_model=True,controlled_epoch_numbers=True,
                controlled_metrics_not_Dice_results=True,epoch_cases=rows,
                parent_error_no_publication=True,disk_full_propagated_and_old_bytes_preserved=True,
                final_cursor=owner.current_epoch,final_sha256=file_hash(final),
                actual_OS_signals_sent=False,signal_termination_not_supported_by_local_parent=True,
                scope='Installed parent normal/error boundaries; not a graceful signal shutdown test')
