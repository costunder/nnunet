"""Deterministic patient minibatches; never trim observations within a patient.

Query scheduling, reference memory, ranking candidates and evaluation are unchanged.
Selections use inner-train metadata only, before any model scoring.
"""
import random
import torch

POLICY = 'patient_episode_v1'


def contract(patients):
    if type(patients) is not int or patients < 2:
        raise ValueError('Explicit support patient count >= 2 required; no automatic reduction')
    return dict(policy=POLICY, patients=patients,
                observations='all eligible rows of selected patients',
                exclusion='query patient on both recipient and donor sides',
                selection='class-covered least-used patients; seeded epoch tie order',
                refresh='one selection and frozen cluster plan per query-patient episode',
                evaluation='full eligible inner-train support, once per query patient')


class PatientEpisodes:
    def __init__(self, rows, query_order, patients, seed, epoch):
        self.contract = contract(patients)
        self.rows = rows
        flat = [i for batch in query_order for i in batch]
        if sorted(flat) != list(range(len(rows))):
            raise ValueError('Query schedule must cover every observation exactly once')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate observation identity')
        for r in rows:
            if r['target'] not in (0, 1) or not r['patient_group'] or not r['donor_group']:
                raise ValueError('Explicit patient/donor identity and binary observation required')
        query_groups = []
        for batch in query_order:
            groups = {rows[i]['patient_group'] for i in batch}
            if len(groups) != 1:
                raise ValueError('Query batch must belong to one patient')
            g = next(iter(groups))
            if not query_groups or query_groups[-1] != g:
                query_groups.append(g)
        if len(set(query_groups)) != len(query_groups):
            raise ValueError('Patient query batches must be contiguous episodes')
        names = sorted({r['patient_group'] for r in rows})
        random.Random(seed + epoch).shuffle(names)
        tie = {g: i for i, g in enumerate(names)}
        uses = dict.fromkeys(names, 0)
        self.selections = {}
        coverage = set()
        for query in query_groups:
            eligible = {}
            for i, r in enumerate(rows):
                if query not in (r['patient_group'], r['donor_group']):
                    eligible.setdefault(r['patient_group'], []).append(i)
            if len(eligible) < patients:
                raise ValueError(f'{query}: requested {patients} support patients, only {len(eligible)} eligible')
            ranked = sorted(eligible, key=lambda g: (uses[g], tie[g]))
            labels = {g: {rows[i]['target'] for i in eligible[g]} for g in ranked}
            selected = []
            missing = {0, 1}
            while missing:
                choices = [g for g in ranked if g not in selected and labels[g] & missing]
                if not choices:
                    raise ValueError(f'{query}: eligible support lacks observed class {sorted(missing)}')
                g = choices[0]
                selected.append(g)
                missing -= labels[g]
            selected += [g for g in ranked if g not in selected][:patients - len(selected)]
            selected = sorted(selected)
            indices = sorted(i for g in selected for i in eligible[g])
            self.selections[query] = dict(patients=selected, indices=indices)
            coverage.update(indices)
            for g in selected:
                uses[g] += 1
        self.audit = dict(policy=self.contract, epoch=epoch,
                          query_observations=len(flat), query_coverage=1.0,
                          support_unique_observations=len(coverage), total_observations=len(rows),
                          support_observation_coverage=len(coverage) / len(rows),
                          support_patient_episode_counts=uses,
                          episodes={g: dict(patients=v['patients'], records=len(v['indices']))
                                    for g, v in self.selections.items()})

    def bind(self, memory):
        """Validate the ordered memory once, then gather only selected rows."""
        if memory['record_ids'] != [r['id'] for r in self.rows]:
            raise ValueError('Support memory record order differs from episode schedule')
        owners = memory['owners'].detach().cpu().tolist()
        classes = memory['classes'].detach().cpu().tolist()
        if len(owners) != len(self.rows) or len(classes) != len(self.rows):
            raise ValueError('Support memory metadata length mismatch')
        if memory['embeddings'].shape != (len(self.rows), 128):
            raise ValueError('Complete 128D reference memory required')
        for i, r in enumerate(self.rows):
            if (memory['patient_groups'][owners[i]] != r['patient_group'] or
                    memory['donor_groups'][i] != r['donor_group'] or classes[i] != r['target']):
                raise ValueError('Support memory patient/donor/class binding differs')
        if memory['embeddings'].requires_grad:
            raise ValueError('Epoch reference memory must be detached')
        self.memory = memory
        self.current = None
        self.current_support = None
        return self

    def support(self, query):
        if self.current != query:
            selection = self.selections[query]
            device = self.memory['embeddings'].device
            ids = torch.tensor(selection['indices'], device=device, dtype=torch.long)
            lookup = {g: i for i, g in enumerate(selection['patients'])}
            owners = torch.tensor([lookup[self.rows[i]['patient_group']] for i in selection['indices']],
                                  device=device, dtype=torch.long)
            self.current_support = (self.memory['embeddings'].index_select(0, ids), owners,
                                    self.memory['classes'].index_select(0, ids))
            self.current = query
        return self.current_support


def verify_migration(previous, current):
    """Explicit learning-policy change from a pinned release; never exact resume."""
    from .execution_upgrade import blob_hash
    if 'support_training' in previous:
        raise ValueError('Already episodic; use exact resume, not a second migration')
    if current.get('support_training') != contract(current.get('support_training', {}).get('patients')):
        raise ValueError('Unknown target support contract')
    if {k:v for k,v in current.items() if k not in ('source','support_training')} != {
            k:v for k,v in previous.items() if k != 'source'}:
        raise ValueError('Support migration cannot change batch/config/precision/resources/activation policy')
    a, b = previous['source'], current['source']
    if a['core'] != b['core']:
        raise ValueError('Support migration cannot change core model/graph/loss')
    allowed = {'l0_regions/training.py', 'l0_regions/support_episodes.py',
               'l0_regions/final.py', 'tools/run_fixed_regions.py'}
    helper = 'l0_regions/execution_upgrade.py'
    if a['runtime'].get(helper) != b['runtime'].get(helper):
        if b['runtime'].get(helper) != blob_hash('d05e4e3', helper):
            raise ValueError('Execution migration helper differs from its reviewed release')
        allowed.add(helper)
    if set(a['runtime']) - set(b['runtime']) or any(
            a['runtime'].get(k) != v for k,v in b['runtime'].items() if k not in allowed):
        raise ValueError('Unreviewed runtime change in support migration')
    for revision in ('d05e4e3', 'f36518c', '87eafa6'):
        if all(v == blob_hash(revision,k) for k,v in a['runtime'].items()):
            return revision
    raise ValueError('Old support runtime is not a pinned reviewed release')
