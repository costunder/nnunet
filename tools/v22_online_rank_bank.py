"""On-demand paired rank -> eligibility -> exact raw/native paste bridge.

No candidate graph cache is published. The reviewed training modules are unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from hiercp_v222.contracts import read_json, write_new, sha, validate_native, validate_identities
from hiercp_v222.bank import EntryBuilder as LegacyBuilder, entry_name, map_center
from tools.v22_online_selection import CONTRACT, selection_from_report, validate_selection


def publish_receipt(path,value):
    """Readers see either no receipt or a complete one; never an open JSON write."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.rank-receipt-',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.link(temp,path)  # Atomic publication; refuses an existing result.
    finally:
        os.unlink(temp)  # Only this call's newly created temporary file.


def online_identity():
    names = ('tools/v22_online_selection.py', 'tools/v22_online_rank_bank.py',
        'tools/v22_online_rank_adapter.py', 'custom_trainers/nnUNetTrainer_OnlineRankV22.py',
        'tools/train_v22_online_rank.py',
        'comparison_randomness.py', 'custom_trainers/nnUNetTrainer_OnlinePairedCP.py',
        'custom_trainers/onlinecp_raw_resampling.py', 'custom_trainers/onlinecp_raw_bank.py',
        'tools/online_raw_bank_preparation.py')
    return {p: sha(ROOT/p) for p in names}


def validate_catalog(meta):
    from hiercp_v22.donors import validate_pool
    from hiercp_v222.v1_cache import provenance
    if (meta.get('selection') != CONTRACT or meta.get('online_identity') != online_identity()
            or meta.get('source_identity') != provenance() or meta.get('complete') is not True):
        raise ValueError('Ranking catalog/source contract mismatch')
    validate_identities(meta['identities'], meta['split'])
    validate_pool(meta['donor_pool'], meta['split'])
    if meta.get('candidate_count') != 128 or meta.get('cp_probability') != .8:
        raise ValueError('Approved CP scale changed')
    expected = {c: [entry_name(c, i) for i in range(len(meta['donor_pool']))]
                for c in meta['split']['outer_train']}
    if meta['entries_by_case'] != expected:
        raise ValueError('Incomplete recipient/donor catalog')
    if sha(meta['checkpoint']) != meta['checkpoint_sha256'] or sha(meta['native_preparation']) != meta['native_sha256']:
        raise ValueError('Changed ranking checkpoint/native preparation')
    return meta


def build_catalog(native_path, checkpoint, output):
    from tools.v22_artifacts import validate_artifact
    from hiercp_v222.v1_cache import provenance
    from hiercp_v22.donors import reject_cross_split_duplicates
    native = validate_native(read_json(native_path))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_artifact(payload, 'final')
    if native['split'] != payload['split']:
        raise ValueError('Native/GNN split mismatch')
    if set(native['planning_patient_ids']) != set(native['split']['outer_train']):
        raise ValueError('Held-out or incomplete native planning cohort')
    if 'identities' in native and native['identities'] != payload['identities']:
        raise ValueError('Native/GNN patient identity mismatch')
    reject_cross_split_duplicates(native['raw_records'], native['split'])
    raw = {r['case_id']: r for r in native['raw_records']}
    for row in payload['raw_records']:
        for key in ('image', 'label'):
            if row[key+'_sha256'] != raw[row['case_id']][key+'_sha256']:
                raise ValueError('Native/GNN raw content mismatch')
    for case in native['split']['outer_train']:
        for key in ('image', 'label'):
            if sha(raw[case][key]) != raw[case][key+'_sha256']:
                raise ValueError('Raw native input changed')
    pool = payload['donor_pool']
    entries = {c: [entry_name(c, i) for i in range(len(pool))] for c in native['split']['outer_train']}
    slots = {c: [dict(source_component=i+1, status='ok', entry=e) for i, e in enumerate(es)] for c, es in entries.items()}
    meta = dict(format='hiercp_online_bank_v2', complete=True, debug=False, selection=CONTRACT,
        completion_scope='catalog only; actual events are scored and filtered on demand',
        paste_contract='onlinecp_raw_target_paste_v1', entries_by_case=entries,
        source_slots_by_case=slots, eligible_sources_by_case={c:list(range(1,len(pool)+1)) for c in entries},
        no_placement_policy='retain_original', source_schedule_format='onlinecp_all_source_slots_v1',
        eligible_source_slots=sum(map(len,slots.values())), no_placement_sources=0,
        candidate_count=128, hier_top_k=1, tumor_label=2, liver_label=1, cp_probability=.8,
        intensity_scale_range=[.95,1.05], intensity_shift_range_hu=[-5.,5.],
        checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha(checkpoint),
        native_preparation=str(Path(native_path).resolve()), native_sha256=sha(native_path),
        split=payload['split'], identities=payload['identities'], donor_pool=pool,
        source_identity=provenance(), online_identity=online_identity(),
        self_patient_policy='keep_original_without_redraw', entry_sha256={})
    validate_catalog(meta)
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    write_new(root/'index.json', meta)
    return root/'index.json'


def proposal_centers(case, source, cfg, base, depth, donor):
    """Existing geometric proposal rules, with no tumor-specific exclusion."""
    from hiercp.common import build_candidate_pool, stable_case_seed
    organ = np.isin(case.label, [1,2]); settings = base['generation']; audit = {}
    pool, _ = build_candidate_pool(case, source, placement_mask=organ, full_organ_mask=organ,
        occupied_mask=np.zeros_like(organ), organ_distance=depth,
        rng=np.random.default_rng(stable_case_seed(cfg['seed'], case.paths.case_id, f'{donor}:{source.component_id}')),
        num_candidates=cfg['candidate_count'], max_draws=settings['max_draws'],
        min_liver_coverage=settings['min_liver_coverage'], occupied_clearance_vox=0,
        min_center_separation_mm=settings['min_center_separation_mm'],
        min_center_separation_vox=settings['min_center_separation_vox'], diagnostics=audit)
    if len(pool) != cfg['candidate_count']:
        raise ValueError('Incomplete geometric proposal pool; no silent reduction')
    return [tuple(p.center) for p in pool], audit


class RankedEntryBuilder(LegacyBuilder):
    # Only the verified raw/native preparation method is inherited. Neither the
    # legacy scorer, proposal prefilter nor argmax materializer is called.
    def __init__(self, index_path, gpu_lock):
        from tools.v22_rank_recommendation import load_checkpoint
        from hiercp.common import discover_cases
        from hiercp_v22.volumes import VolumeCache
        from custom_trainers.onlinecp_raw_bank import RawBankStore
        self.path=Path(index_path).resolve(); self.root=self.path.parent
        self.meta=validate_catalog(read_json(self.path)); self.native=validate_native(read_json(self.meta['native_preparation']))
        # Frozen scorer construction must not consume segmentation RNG state.
        with gpu_lock,torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            self.network,self.memory,self.payload=load_checkpoint(self.meta['checkpoint'])
        if (self.payload['split'] != self.native['split'] or self.payload['donor_pool'] != self.meta['donor_pool']
                or self.payload['identities'] != self.meta['identities']):
            raise ValueError('Native/ranking donor, split or identity mismatch')
        self.cfg,self.base=self.payload['config'],self.payload['base']; self.gpu_lock=gpu_lock
        all_paths={p.case_id:p for p in discover_cases(Path(self.native['medical'])/'Data')}
        self.paths={c:all_paths[c] for c in self.meta['split']['outer_train']}
        raw={r['case_id']:r for r in self.native['raw_records']}
        for c,p in self.paths.items():
            if sha(p.image_path)!=raw[c]['image_sha256'] or sha(p.label_path)!=raw[c]['label_sha256']:
                raise ValueError('Original CT/GT changed')
        self.volumes=VolumeCache(self.paths,self.base,self.cfg)
        self.plans=read_json(self.native['plans']); self.plan=self.plans['configurations']['3d_fullres']
        self.locks={c:threading.RLock() for c in self.paths};self.entry_locks={};self.lock=threading.Lock()
        self.store=RawBankStore(self.root);self.raw_ready={};self.native_lock=threading.Lock()
        # Serialize full event graph inventories; CPU graph tasks still run in
        # measured parallel waves. Avoid N simultaneous 128-graph RAM inventories.
        self.event_lock=threading.Lock()

    def materialize(self, recipient, donor_index):
        from hiercp_v222.v1_local import prepare_donor, pair_record
        from hiercp_v22.data import donor_in_target_spacing
        from hiercp_v222.placement import placement_spec
        from tools.v22_rank_recommendation import recommend
        from tools.online_raw_bank_preparation import prepare_source_candidates
        from hiercp.preparation_runtime import run_case_jobs
        if recipient not in self.meta['split']['outer_train'] or type(donor_index)!=int or not 0<=donor_index<len(self.meta['donor_pool']):
            raise ValueError('Recipient/donor outside training catalog')
        name=entry_name(recipient,donor_index); receipt=self.root/(name+'.receipt.json')
        with self.event_lock:
            if receipt.exists():
                value=read_json(receipt)
                if value['catalog_sha256']!=sha(self.path):raise ValueError('Stale event receipt')
                return dict(relative=name)
            row=self.meta['donor_pool'][donor_index]; donor=row['case_id']
            value=dict(catalog_sha256=sha(self.path),recipient=recipient,donor=row,relative=name)
            identities=self.meta['identities']['cases']; group=identities[recipient]['patient_group']
            if identities[donor]['patient_group']==group:
                value.update(status='self_patient_no_placement',selection=None)
            else:
                with self.volumes.pair(recipient,donor) as (target,donor_data):
                    case=target['case'];dc=donor_data['case'];collection=donor_data['sources']
                    pos=[c for c,_ in collection.entries].index(row['component_id']);source,diameter=collection[pos]
                    prepared=prepare_donor(dc,source,donor_data['organ'],donor_data['depth'],self.base)
                    transformed,_=donor_in_target_spacing(source,dc.spacing,case.spacing)
                    centers,audit=proposal_centers(case,transformed,self.cfg,self.base,target['depth'],donor)
                    placements=[placement_spec(case,transformed,c,donor) for c in centers]
                    records={}
                    def graph(i):
                        p=placements[i]
                        return i,pair_record(case,source,dc.spacing,prepared,p.center,target['organ'],target['depth'],self.base,donor_id=donor,placement=p)
                    run_case_jobs(tasks=list(range(len(centers))),function=graph,
                        commit=lambda r:records.__setitem__(*r),workers='auto',
                        report_path=self.root/f'resources/{recipient}_{donor_index}.json')
                    with self.gpu_lock:
                        report=recommend(self.network,[records[i] for i in range(len(centers))],self.memory,
                            query_group=group,batch_size=self.payload['physical_batch'],workers=self.payload['workers'],
                            recipient_case=case,placements=placements,min_liver_coverage=self.base['generation']['min_liver_coverage'])
                    del records
                    selection=selection_from_report(report,placements);chosen=validate_selection(selection)
                    value.update(status='ranked',selection=selection,proposal_audit=audit,diameter=diameter)
                    if chosen is not None:
                        p=placements[chosen]
                        native,ref,digest,props=self._native_case(recipient,case)
                        mapped=np.stack([map_center(c,self.plans,props,native['metadata']['preprocessed_shape']) for c in centers])
                        refs,digests,transport=prepare_source_candidates(self.root,recipient,donor_index+1,native,digest,
                            p.image,p.mask,p.anchor,np.asarray([p.center]),self.plan['patch_size'])
                        value.update(native_centers=mapped.tolist(),raw_case_reference=ref,raw_case_reference_sha256=digest,
                            selected_payload=str(refs[0]),selected_payload_sha256=str(digests[0]),
                            placement=p.metadata(),transport_audit=transport)
            publish_receipt(receipt,value)
            return dict(relative=name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(build_catalog(args.native,args.checkpoint,args.output),flush=True)
