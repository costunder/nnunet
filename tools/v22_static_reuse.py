"""Explicit execution candidate: cache fixed inputs, never learned outputs.

Not installed by the production entry point. The bounded comparison tool can
install this on the same model/checkpoint without changing research settings.
Only the last support topology, query topology and case reference are retained.
Tensor mutation counters invalidate caches; unsafe .data/external storage writes
are outside the training loop's immutable-memory contract.
"""
from contextlib import contextmanager, ExitStack
from types import MethodType
from unittest.mock import patch
import time

import torch
from torch.utils.checkpoint import checkpoint


def signature(value):
    """Inspect metadata/versions without copying tensor contents off the GPU."""
    if torch.is_tensor(value):
        return ('tensor', id(value), value._version, value.dtype, value.device,
                tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.data_ptr())
    if isinstance(value, dict):
        return ('dict', tuple((k, signature(v)) for k, v in sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(signature(v) for v in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return (type(value).__name__, value)
    raise TypeError('Unsupported immutable execution state: ' + str(type(value)))


class StaticGraphs:
    def __init__(self):
        self.support = self.query = self.checked_plan = None
        self.support_builds = self.query_builds = 0

    def check_plan(self, plan, embeddings, owners, classes):
        values = (plan['owners'], plan['classes'], plan['support_embeddings'], owners, classes, embeddings)
        key = signature(values)
        if self.checked_plan is not None and self.checked_plan[0] == key:
            return
        if not torch.equal(plan['owners'], owners) or not torch.equal(plan['classes'], classes):
            raise ValueError('Cluster plan and explicit support ownership/classes differ')
        if not torch.equal(plan['support_embeddings'], embeddings):
            raise ValueError('Cluster plan was fitted on a different support memory')
        self.checked_plan = key, values

    def topology(self, embeddings, owners, classes):
        if embeddings.ndim != 2 or owners.shape != classes.shape or owners.shape != (len(embeddings),):
            raise ValueError('Malformed explicit support')
        if owners.dtype != torch.long or classes.dtype != torch.long or not len(owners):
            raise ValueError('Support owner/class int64 IDs required')
        key = (signature(owners), signature(classes), len(embeddings), embeddings.device)
        if self.support is not None and self.support[0] == key:
            return self.support[2]
        p = int(owners.max()) + 1
        if p < 2 or bool((torch.bincount(owners, minlength=p) == 0).any()):
            raise ValueError('At least two nonempty other-patient support tasks required')
        if bool(((classes < 0) | (classes > 1)).any()) or len(classes.unique()) != 2:
            raise ValueError('Both observed classes must exist in training support')
        n = len(embeddings); device = embeddings.device
        data = torch.arange(n, device=device).repeat_interleave(2)
        cls = torch.arange(2, device=device).repeat(n)
        label = n + owners.repeat_interleave(2) * 2 + cls
        known = torch.stack((torch.ones_like(cls), (classes.repeat_interleave(2) == cls).long()), -1).float()
        edges = torch.stack((torch.cat((data, label)), torch.cat((label, data))))
        features = torch.cat((known, known))
        task = torch.arange(p, device=device).repeat_interleave(2)
        forbidden = task[:, None] == task[None, :]
        value = (p, edges, features, forbidden)
        # Keep inputs alive so a recycled Python object ID cannot hit this entry.
        self.support = key, (owners, classes), value
        self.support_builds += 1
        return value

    def query_topology(self, q, labels):
        key = (len(labels), len(q), q.device, q.dtype)
        if self.query is None or self.query[0] != key:
            src = torch.arange(len(labels), device=q.device).repeat(len(q))
            dst = torch.arange(len(q), device=q.device).repeat_interleave(len(labels))
            self.query = key, (src, dst, q.new_zeros(len(src), 2))
            self.query_builds += 1
        return self.query[1]


@contextmanager
def installed_model(net):
    """Keep every trainable pass, dropout draw and checkpoint in original order."""
    from hiercp_v222.model import alignment_loss, prototype_logits
    cache = StaticGraphs()

    def encode(self, embeddings, owners, classes):
        p, edges, features, _ = cache.topology(embeddings, owners, classes)
        n = len(embeddings)
        x = torch.cat((embeddings, self.label_seed.repeat(p, 1)))
        histories = []
        for layer in self.l1:
            histories.append(x[n:])
            x = checkpoint(layer, x, edges, features, use_reentrant=False) if self.training else layer(x, edges, features)
        return histories, x[n:].reshape(p, 2, self.dim)

    def prepare(self, embeddings, owners, classes, cluster_plan=None):
        plan = self.fit_support_clusters(embeddings, owners, classes) if cluster_plan is None else cluster_plan
        cache.check_plan(plan, embeddings, owners, classes)
        histories, local_labels = self.encode_support(embeddings, owners, classes)
        p = len(local_labels)
        forbidden = cache.topology(embeddings, owners, classes)[3]
        aligned = local_labels.flatten(0, 1)[None]
        for layer, update in zip(self.l2, self.l2_updates):
            def run(value, layer=layer, update=update):
                return update(value, layer(value, value, value, attn_mask=forbidden, need_weights=False)[0])
            aligned = checkpoint(run, aligned, use_reentrant=False) if self.training else run(aligned)
        labels = aligned[0].reshape(p, 2, self.dim)
        return dict(histories=histories, labels=labels, local_labels=local_labels,
                    cluster_plan=plan, alignment_loss=alignment_loss(labels, plan, self.temperature))

    def predict(self, query, state):
        q = query
        for layer, labels in zip(self.l1, state['histories']):
            src, dst, edge = cache.query_topology(q, labels)
            q = layer.messages(labels, q, src, dst, edge)
        logits = prototype_logits(q, state['labels'], state['cluster_plan'], self.temperature)
        return {'logits': logits, 'ranking_score': logits.softmax(-1)[:, 1],
                'alignment_loss': state['alignment_loss'], 'alignment_loss_weight': self.alignment_loss_weight}

    with ExitStack() as stack:
        for name, function in (('encode_support', encode), ('prepare_support', prepare), ('predict_embeddings', predict)):
            stack.enter_context(patch.object(net, name, MethodType(function, net)))
        yield cache


@contextmanager
def installed_reference(context):
    original = context.reference
    slot = None
    builds = {'reference_builds': 0}

    def reference(ids, device):
        nonlocal slot
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicated live ranking query')
        rows = context.dataset.rows
        selected = tuple(sorted({rows[i]['case_id'] for i in ids}))
        key = (selected, torch.device(device), signature(context.memory['embeddings']))
        if slot is None or slot[0] != key:
            refs, _, observed, cases = original(ids, device)
            all_ids = [i for case in selected for i in context.by_case[case]]
            positions = {index: j for j, index in enumerate(all_ids)}
            slot = key, context.memory['embeddings'], refs, observed, cases, positions
            builds['reference_builds'] += 1
        _, _, refs, observed, cases, positions = slot
        locations = torch.tensor([positions[i] for i in ids], device=device, dtype=torch.long)
        return refs, locations, observed, cases

    with patch.object(context, 'reference', reference):
        yield builds


class FrozenFields:
    """One CPU snapshot/hash per unchanged state field; never cache current weights."""
    names = ('memory', 'memory_generation', 'plan', 'plan_generation')

    def __init__(self):
        self.entries = {}
        self.hits = self.misses = 0
        self.seconds = dict(signature=0., copy=0., hash=0.)

    def get(self, name, value):
        from tools.v22_artifacts import tree_hash
        began = time.perf_counter()
        key = signature(value)
        self.seconds['signature'] += time.perf_counter() - began
        old = self.entries.get(name)
        if old is not None and old[0] == key:
            self.hits += 1
            return old[2], old[3]
        # Importing backend.snapshot here can recurse while installed: use the
        # preserved function captured below, not the temporarily patched binding.
        began = time.perf_counter()
        frozen = _snapshot(value)
        self.seconds['copy'] += time.perf_counter() - began
        began = time.perf_counter()
        digest = tree_hash(frozen)
        self.seconds['hash'] += time.perf_counter() - began
        self.entries[name] = (key, value, frozen, digest)
        self.misses += 1
        return frozen, digest


from tools.v222_runtime_execution import snapshot as _snapshot
from tools.v222_support_snapshot import AsyncSaver as _Saver


class ReuseSaver(_Saver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_fields = FrozenFields()
        self.static_digests = {}
        self.last_reuse_metrics = {}

    def close(self):
        try:
            super().close()
        finally:
            self.static_fields.entries.clear()
            self.static_digests.clear()

    def freeze(self, value, memo=None):
        if not isinstance(value, dict) or not {'format', 'model', 'optimizer', 'state', 'rng'} <= value.keys():
            return _snapshot(value, memo)
        state = value['state']
        shared = {}
        self.static_digests = {}
        for name in self.static_fields.names:
            frozen, digest = self.static_fields.get(name, state.get(name))
            if name in state:
                shared[name] = frozen
            self.static_digests[name] = digest
        rest = dict(value, state={k: v for k, v in state.items() if k not in shared})
        if self.frozen_payload is not None:
            if state['phase'] not in ('initial_memory', 'refresh_memory', 'final_memory'):
                raise RuntimeError('Fixed snapshot cannot be used for an optimizer update')
            if self.versions() != self.frozen_versions:
                raise RuntimeError('Model changed inside no-update support pass')
            rest = {k: v for k, v in rest.items() if k not in ('model', 'optimizer')}
        result = _snapshot(rest, memo)
        result['state'].update(shared)
        if self.frozen_payload is not None:
            result.update(self.frozen_payload)
        return result

    def seal(self, payload):
        from tools.v22_artifacts import tree_hash
        from tools.v22_resume_integrity import rng_hash
        began = time.perf_counter()
        payload['model_sha256'] = tree_hash(payload['model'])
        payload['rng_sha256'] = rng_hash(payload['rng'])
        payload['resume_integrity'] = dict(self.static_digests,
            **{k: tree_hash(payload['state'].get(k)) for k in ('memory_work', 'memory_work_generation')})
        self.seal_seconds = time.perf_counter() - began

    def save(self, state):
        from tools import v222_runtime_execution as backend
        from tools import v22_resume_integrity as integrity
        before = self.static_fields.hits, self.static_fields.misses
        times = self.static_fields.seconds.copy()
        self.seal_seconds = 0.
        start = time.perf_counter()
        with patch.object(backend, 'snapshot', self.freeze), patch.object(integrity, 'seal_resume', self.seal):
            result = super().save(state)
        self.last_reuse_metrics = dict(seconds=time.perf_counter()-start,
            reused_fields=self.static_fields.hits-before[0], copied_fields=self.static_fields.misses-before[1],
            seal_seconds=self.seal_seconds,
            fixed_fields_seconds={key: self.static_fields.seconds[key]-times[key] for key in times})
        return result
