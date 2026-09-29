"""Explicit, validated reuse of completed batches; never rewrite their provenance."""
import copy
import ast
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from .data import load
from .preparation import PREPARATION_FILES, _binding
from tools.v22_artifacts import tree_hash

ROOT=Path(__file__).resolve().parents[1]
# Reviewed revisions share the single-scale partition/model/profile. Their
# differences are packaging, verified recovery and frozen-CNN loading.
COMPATIBLE_REVISIONS=('6be85aa836833d7a61d590a81d9d3eb41e267504',
                      '6d5f0dc9bbcdf5bd4261887848f439dbaac233d9',
                      '72be3cec8a8a5effaf2b318f692267901b1c93e3',
                      'f07b13f5cc656aa413fc1f89b4f66c0925158559',
                      '27b598b1ec5a8738737c477e1dd8a9953470fa70',
                      'b398b3d8c0044d883729b789e83d69c0ed841725')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=len(COMPATIBLE_REVISIONS))
def revision_identity(revision):
    """Only immutable pinned Git blobs are cached, never live source files."""
    def read(name):
        return subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',
            'show',f'{revision}:{name}'],cwd=ROOT)
    source=read('l0_regions/preparation.py')
    declarations=[node for node in ast.parse(source).body if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='PREPARATION_FILES' for t in node.targets)]
    names=set(ast.literal_eval(declarations[0].value))|{'l0_regions/training_data.py'}
    return {name:hashlib.sha256(source if name=='l0_regions/preparation.py' else read(name)).hexdigest() for name in names}


def verified_origin(identity,current):
    if identity==current:return 'same_preparation_source'
    from .execution_upgrade import compatible_core
    if not compatible_core(identity.get('core',{}),current['core']):raise ValueError('Reuse core implementation differs')
    names=set(identity.get('preparation',{}))
    reviewed={'l0_regions/preparation.py','l0_regions/training_data.py','l0_regions/preparation_reuse.py','l0_regions/preparation_runtime.py',
              'l0_regions/data.py'}  # Exact duplicate-edge predicate optimization; no schema/check removed.
    if not names or names-set(current['preparation']):raise ValueError('Unknown reuse preparation file set')
    if any(identity['preparation'][name]!=current['preparation'][name] for name in names-reviewed):
        raise ValueError('Reuse partition/integrity implementation changed')
    for revision in COMPATIBLE_REVISIONS:
        if identity['preparation']==revision_identity(revision):return revision
    raise ValueError('Reuse preparation is not a reviewed compatible revision')


def binding_digest(identity):
    source=dict(identity['core'])
    source.update({k:v for k,v in identity['preparation'].items() if k!='l0_regions/training_data.py'})
    return hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()


def verified_binding_sources(meta,current):
    """Historical source receipts remain intact; only reviewed origins can mix."""
    origins=[meta['source_identity']]
    proof=meta.get('reused_preparation')
    if proof:
        if proof.get('format')!='verified_prepared_batch_reuse_v1' or proof.get('original_items_relabelled') is not False:
            raise ValueError('Unknown reused preparation proof')
        if proof.get('binding_source_sha256')!=binding_digest(proof['source_identity']):
            raise ValueError('Reused binding source digest changed')
        origins.append(proof['source_identity'])
        origins.extend(proof.get('source_origins',[]))
    allowed=set()
    for origin in origins:
        verified_origin(origin,current);allowed.add(binding_digest(origin))
    for entries in meta['partitions'].values():
        for entry in entries:
            if entry['binding']['preparation_source_sha256'] not in allowed:
                raise ValueError('Unreviewed record preparation source')
    return frozenset(allowed)


class PreparedReuse:
    def __init__(self,folder,expected,batch):
        self.folder=Path(folder).resolve()
        self.request_path=self.folder/'request.json';self.request_hash=digest(self.request_path)
        self.request=json.loads(self.request_path.read_text(encoding='utf-8'))
        for key in ('format','debug','original_cache_sha256','partition_checkpoint_sha256',
                    'cnn_sha256','profile','view_epoch','config','base','profile_policy','partition_quality_validated'):
            if self.request.get(key)!=expected[key]:raise ValueError('Reuse request mismatch: '+key)
        revision=verified_origin(self.request['source_identity'],expected['source_identity'])
        # Repeated interruption can leave original and newly created batches in
        # the same folder. Retain each item's ORIGINAL reviewed binding source.
        origins=[self.request['source_identity']]
        proof=self.request.get('reused_preparation')
        if proof:
            if proof.get('format')!='verified_prepared_batch_reuse_v1' or proof.get('original_items_relabelled') is not False:
                raise ValueError('Unknown chained reuse proof')
            prior=proof.get('source_origins',[proof['source_identity']])
            if not isinstance(prior,list) or not prior:raise ValueError('Missing reuse source origins')
            if proof['binding_source_sha256']!=binding_digest(proof['source_identity']):
                raise ValueError('Chained binding source digest changed')
            origins.extend(prior)
        self.allowed_sources={}
        for origin in origins:
            verified_origin(origin,expected['source_identity'])
            self.allowed_sources[binding_digest(origin)]=origin
        self.frozen_hash=digest(self.folder/'frozen_cnn.pt')
        if self.frozen_hash!=self.request['cnn_file_sha256']:raise ValueError('Reuse CNN file changed')
        # This reproduces the source digest of the ORIGINAL item binding. Never
        # stamp the current preparation digest onto previously created tensors.
        self.binding_sha=binding_digest(self.request['source_identity'])
        self.provenance=dict(format='verified_prepared_batch_reuse_v1',path=str(self.folder),
            request_sha256=self.request_hash,revision=revision,source_identity=self.request['source_identity'],
            binding_source_sha256=self.binding_sha,original_items_relabelled=False,
            source_origins=list(self.allowed_sources.values()),
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
            manifest=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            origin_sha=manifest['binding']['preparation_source_sha256']
            if origin_sha not in self.allowed_sources:raise ValueError('Unreviewed item binding source')
            row=dict(ds.rows[i],donor_component=ds.record(i)['component_id'])
            binding=_binding(record=row,dataset_index=i,source_sha=origin_sha,
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
