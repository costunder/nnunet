"""Exact support bindings and sampled evaluation for cloned L1 candidates.

No production model, inventory, memory, plan, or checkpoint is written here.
"""
import torch

from hiercp_v222.v1_local import support_for_recipient
from l0_regions.training import hash_state
from tools.local_cnn_fusion_probe import support_indices
from tools.v22_candidate_order import record_key
from tools.v22_rank_objective import ranking_metrics


def support_binding(memory, rows, query_group, episodes=None):
    if memory['record_ids'] != [row['id'] for row in rows]:
        raise ValueError('Candidate support order differs from the original memory')
    if memory['donor_groups'] != [row['donor_group'] for row in rows]:
        raise ValueError('Candidate donor binding differs from original memory')
    if (memory['embeddings'].ndim!=2 or memory['embeddings'].shape!=(len(rows),128)
            or memory['embeddings'].dtype!=torch.float32
            or memory['owners'].shape!=(len(rows),) or memory['classes'].shape!=(len(rows),)
            or memory['owners'].dtype!=torch.long or memory['classes'].dtype!=torch.long
            or memory['owners'].device!=memory['embeddings'].device
            or memory['classes'].device!=memory['embeddings'].device
            or not bool(torch.isfinite(memory['embeddings']).all())):
        raise ValueError('Finite complete 128D original memory and int64 metadata required')
    owners=memory['owners'].detach().cpu().tolist()
    classes=memory['classes'].detach().cpu().tolist()
    if len(owners)!=len(rows) or len(classes)!=len(rows):
        raise ValueError('Candidate original owner/class coverage differs')
    for row,owner,target in zip(rows,owners,classes):
        if (not 0<=owner<len(memory['patient_groups'])
                or memory['patient_groups'][owner]!=row['patient_group'] or target!=row['target']):
            raise ValueError('Candidate original owner/class observation binding differs')
    if episodes is None:
        indices = support_indices(rows, query_group)
        support = support_for_recipient(memory, query_group)
    else:
        indices = list(episodes.selections[query_group]['indices'])
        selected = set(episodes.selections[query_group]['patients'])
        expected = [i for i, row in enumerate(rows)
                    if row['patient_group'] in selected
                    and query_group not in (row['patient_group'], row['donor_group'])]
        if indices != expected:
            raise ValueError('Candidate episode lost or substituted eligible observations')
        support = episodes.support(query_group)
    ids = torch.tensor(indices, device=memory['embeddings'].device, dtype=torch.long)
    if not torch.equal(support[0], memory['embeddings'].index_select(0, ids)):
        raise ValueError('Candidate support features differ from the bound memory')
    if not torch.equal(support[2], memory['classes'].index_select(0, ids)):
        raise ValueError('Candidate support classes differ from original observation labels')
    selected_groups=(list(episodes.selections[query_group]['patients']) if episodes is not None
                     else [memory['patient_groups'][owner] for owner in sorted({owners[i] for i in indices})])
    mapping={group:i for i,group in enumerate(selected_groups)}
    expected_owners=torch.tensor([mapping[rows[i]['patient_group']] for i in indices],
                                device=support[1].device,dtype=torch.long)
    if not torch.equal(support[1],expected_owners):
        raise ValueError('Candidate compact support owner mapping differs')
    if any(query_group in (rows[i]['patient_group'], rows[i]['donor_group']) for i in indices):
        raise ValueError('Query group leaked into candidate support')
    return support, [rows[i]['id'] for i in indices], query_group


def make_evaluator(cases, memory, train_rows, physical_batch, budget, original_local_hash):
    """All candidates of explicit selected cases; fresh L1 plan per model/case.

    A frozen saved L0 support table is shared across the two diagnostic training
    arms, as within an ordinary optimization epoch. No full memory refresh or
    complete validation cycle is implied. Current query CNN outputs are reused
    only when the entire local state still equals the original snapshot.
    """
    if type(physical_batch) is not int or physical_batch < 2:
        raise ValueError('Original physical batch >=2 required')
    if not cases or {case['split'] for case in cases} != {'train', 'validation'}:
        raise ValueError('Explicit train and validation diagnostic cases required')

    @torch.no_grad()
    def evaluate(model):
        from tools.diagnose_local_cnn_learning import trace_head, score_summary
        modes = [(module, module.training) for module in model.modules()]
        outputs = {split: [] for split in ('train', 'validation')}
        score_parts = {split: [] for split in outputs}
        truth_parts = {split: [] for split in outputs}
        names = {split: [] for split in outputs}
        keys = {split: [] for split in outputs}
        try:
            model.eval()
            reuse = hash_state(model.local.state_dict()) == original_local_hash
            device=next(model.parameters()).device
            for case in cases:
                rows, ids = case['rows'], case['indices']
                if not ids or len(ids) != len(rows) or len({row['case_id'] for row in rows}) != 1:
                    raise ValueError('Complete single-case query binding required')
                if not torch.equal(case['truth'].cpu(), torch.tensor([row['target'] for row in rows])):
                    raise ValueError('Candidate evaluation target binding differs')
                data=case['reader'].ds
                expected=[i for i,row in enumerate(data.rows) if row['case_id']==rows[0]['case_id']]
                if ids!=expected or rows!=[data.rows[i] for i in expected]:
                    raise ValueError('Candidate evaluation lost or substituted original case observations')
                if reuse:
                    embeddings = case['embeddings']
                else:
                    parts = []
                    for start in range(0, len(ids), physical_batch):
                        budget.check()
                        query = case['reader'].get(ids[start:start+physical_batch]).to(device)
                        parts.append(model.local(query).detach())
                        del query
                    embeddings = torch.cat(parts)
                support, record_ids, _ = support_binding(memory, train_rows, rows[0]['patient_group'])
                if embeddings.shape!=(len(rows),model.dim) or embeddings.device!=device:
                    raise ValueError('Candidate evaluation query shape/device differs')
                scores, trace = trace_head(model, embeddings, support, physical_batch)
                truth = case['truth']
                candidate_keys = [record_key(row) for row in rows]
                metrics, details = ranking_metrics(scores.cpu(), truth.cpu(),
                    [rows[0]['case_id']]*len(rows), candidate_keys=candidate_keys)
                score = score_summary(scores, truth)
                split = case['split']
                outputs[split].append(dict(case_id=rows[0]['case_id'], records=len(rows),
                    all_case_candidates_retained=True, metrics=metrics, score=score,
                    observed_ranks=details[0]['observed_ranks'], trace=trace,
                    support_record_ids=record_ids, cached_query_basis_reused=reuse))
                score_parts[split].append(scores.cpu())
                truth_parts[split].append(truth.cpu())
                names[split].extend([rows[0]['case_id']]*len(rows))
                keys[split].extend(candidate_keys)
                budget.check()
            result = {}
            for split in outputs:
                metrics, _ = ranking_metrics(torch.cat(score_parts[split]), torch.cat(truth_parts[split]),
                    names[split], candidate_keys=keys[split])
                comparisons = sum(case['score']['comparisons'] for case in outputs[split])
                result[split] = dict(metrics=metrics,
                    pair_win_rate=sum(case['score']['pair_win_rate']*case['score']['comparisons']
                                      for case in outputs[split])/comparisons,
                    cases=outputs[split])
            result['scope'] = 'all candidates of explicitly selected cases; NOT full validation or CP efficacy'
            result['support_scope'] = 'full eligible saved detached L0 table; own L1 histories and teacher plan'
            return result
        finally:
            for module, mode in modes:
                module.training = mode
    return evaluate
