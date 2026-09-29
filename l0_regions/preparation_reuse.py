"""Explicit, validated reuse of completed batches; never rewrite their provenance."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from .data import load
from .preparation import PREPARATION_FILES, _binding
from tools.v22_artifacts import tree_hash

ROOT=Path(__file__).resolve().parents[1]
# Both revisions use the identical single-scale partition/model/profile. The
# latter only deduplicates receipts and packaging, independently GPU-tested.
COMPATIBLE_REVISIONS=('6be85aa836833d7a61d590a81d9d3eb41e267504',
                      '6d5f0dc9bbcdf5bd4261887848f439dbaac233d9')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_origin(identity,current):
    if identity==current:return 'same_preparation_source'
    if identity.get('core')!=current['core']:raise ValueError('Reuse core implementation differs')
    names=set(PREPARATION_FILES)-{'l0_regions/preparation_reuse.py'}
    names.add('l0_regions/training_data.py')
    if set(identity.get('preparation',{}))!=names:raise ValueError('Unknown reuse preparation file set')
    reviewed={'l0_regions/preparation.py','l0_regions/training_data.py'}
    if any(identity['preparation'][name]!=current['preparation'][name] for name in names-reviewed):
        raise ValueError('Reuse partition/integrity implementation changed')
    for revision in COMPATIBLE_REVISIONS:
        matches=True
        for name in sorted(names):
            raw=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',
                'show',f'{revision}:{name}'],cwd=ROOT)
            if hashlib.sha256(raw).hexdigest()!=identity['preparation'][name]:
                matches=False;break
        if matches:return revision
    raise ValueError('Reuse preparation is not a reviewed compatible revision')


class PreparedReuse:
    def __init__(self,folder,expected,batch):
        self.folder=Path(folder).resolve()
        self.request_path=self.folder/'request.json';self.request_hash=digest(self.request_path)
        self.request=json.loads(self.request_path.read_text(encoding='utf-8'))
        for key in ('format','debug','original_cache_sha256','partition_checkpoint_sha256',
                    'cnn_sha256','profile','view_epoch','config','base','profile_policy','partition_quality_validated'):
            if self.request.get(key)!=expected[key]:raise ValueError('Reuse request mismatch: '+key)
        if self.request.get('reused_preparation'):raise ValueError('Chained cross-version reuse requires separate review')
        revision=verified_origin(self.request['source_identity'],expected['source_identity'])
        self.frozen_hash=digest(self.folder/'frozen_cnn.pt')
        if self.frozen_hash!=self.request['cnn_file_sha256']:raise ValueError('Reuse CNN file changed')
        # This reproduces the source digest of the ORIGINAL item binding. Never
        # stamp the current preparation digest onto previously created tensors.
        source=dict(self.request['source_identity']['core'])
        source.update({k:v for k,v in self.request['source_identity']['preparation'].items()
                       if k!='l0_regions/training_data.py'})
        self.binding_sha=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
        self.provenance=dict(format='verified_prepared_batch_reuse_v1',path=str(self.folder),
            request_sha256=self.request_hash,revision=revision,source_identity=self.request['source_identity'],
            binding_source_sha256=self.binding_sha,original_items_relabelled=False,
            exact_future_partition_rng_resume=False)
        self.batch=batch;self.used_audits={};self.reused_records=0;self.reused_batches=0
        # A completed audit is the old writer's final batch marker. Require
        # its whole original batch, never reinterpret it as a smaller batch.
        self.audits={}
        for part in ('inner_train','inner_val'):
            self.audits[part]={}
            for path in self.folder.glob(part+'_*_audit.json'):
                start=int(path.name[len(part)+1:-len('_audit.json')])
                if start%batch:raise ValueError('Reuse preparation batch schedule differs')
                self.audits[part][start]=path

    def restore(self,part,ids,ds,pool):
        audit_path=self.audits[part].get(ids[0])
        if audit_path is None:return None
        audit_hash=digest(audit_path)
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        if audit.get('scale2')!={'status':'REMOVED_BY_DESIGN'}:raise ValueError('Reuse requires single-scale audit')
        paths=[self.folder/f'{part}_{i:06d}.pt' for i in ids]
        if any(not p.is_file() or not p.with_suffix('.json').is_file() for p in paths):
            raise ValueError('Completed batch audit has missing item files')
        groups=audit['scale1']['roles']
        owners={g['owner'] for role in groups.values() for g in role['group_diagnostics']}
        if owners!=set(range(len(ids))):raise ValueError('Reuse batch coverage/size differs')
        def one(arg):
            position,i,path=arg
            row=dict(ds.rows[i],donor_component=ds.record(i)['component_id'])
            binding=_binding(record=row,dataset_index=i,source_sha=self.binding_sha,
                cache_sha256=self.request['original_cache_sha256'],frozen_cnn_sha256=self.request['cnn_sha256'],
                profile=self.request['profile'],view_epoch=self.request['view_epoch'],view_index=0,
                feature_evidence='checkpoint_partition_quality_unverified')
            item=load(path,binding)
            if item['preparation_pair_index']!=position:raise ValueError('Reuse item owner mismatch')
            if item['preparation_batch_profile_exceeded']!=audit['scale1']['profile_exceeded']:
                raise ValueError('Reuse batch admission mismatch')
            embedded=item['audit']
            if embedded.get('format')=='region_pair_audit_v1':
                if embedded['batch_audit_sha256']!=tree_hash(audit):raise ValueError('Reuse batch audit digest mismatch')
            elif tree_hash(embedded)!=tree_hash(audit):raise ValueError('Reuse full batch audit differs')
            for role,stats in groups.items():
                expected=[g for g in stats['group_diagnostics'] if g['owner']==position]
                actual=[g for g in embedded['scale1']['roles'][role]['group_diagnostics'] if g['owner']==position]
                if actual!=expected:raise ValueError('Reuse own-group diagnostics differ')
            return item
        items=list(pool.map(one,zip(range(len(ids)),ids,paths)))
        for key in ('materialized_batch_sha256','fine_edges_sha256'):
            if len({v['fine_graph_evidence'][key] for v in items})!=1:raise ValueError('Reuse mixed materialization batch')
        if digest(audit_path)!=audit_hash:raise ValueError('Reuse audit changed during validation')
        self.used_audits[str(audit_path)]=audit_hash
        self.reused_records+=len(ids);self.reused_batches+=1
        return items,audit

    def finish(self):
        if digest(self.request_path)!=self.request_hash or digest(self.folder/'frozen_cnn.pt')!=self.frozen_hash:
            raise ValueError('Reuse source request/CNN changed')
        for path,expected in self.used_audits.items():
            if digest(path)!=expected:raise ValueError('Reuse source audit changed')
        return dict(self.provenance,reused_records=self.reused_records,reused_batches=self.reused_batches,
                    batch_audit_sha256=copy.deepcopy(self.used_audits))
