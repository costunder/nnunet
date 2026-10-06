"""Training-only stage gate on fresh full-bank L0 features; never validation GT."""
from collections import defaultdict
import time
import torch
from tqdm import tqdm
from hiercp_v222.v1_execution import rng_state, restore_rng
from hiercp_v222.v1_local import support_for_recipient


@torch.no_grad()
def evaluate_gate(net, ds, memory, curriculum):
    """Score each current training case jointly, reusing its just-refreshed L0.

    Labels are read only AFTER predicting scores. Query labels never enter the
    model. Support keeps every eligible training row, including deferred U.
    """
    if memory['record_ids'] != [r['id'] for r in ds.rows]:
        raise ValueError('Curriculum gate requires the ordered full training bank')
    if memory['embeddings'].shape != (len(ds.rows), 128) or memory['embeddings'].requires_grad:
        raise ValueError('Fresh detached full 128D L0 memory required')
    grouped = defaultdict(list)
    for i in curriculum.active_indices():
        grouped[ds.rows[i]['case_id']].append(i)
    saved_rng = rng_state()
    modes = [(m, m.training) for m in net.modules()]
    cases = []
    totals = []
    net.eval()
    try:
        for case, ids in tqdm(sorted(grouped.items()), desc='TRAIN curriculum gate', unit='case'):
            start=time.perf_counter()
            owners = {ds.rows[i]['patient_group'] for i in ids}
            if len(owners) != 1:
                raise ValueError('Gate case identity differs')
            index = torch.tensor(ids, device=memory['embeddings'].device)
            query = memory['embeddings'].index_select(0, index)
            support = support_for_recipient(memory, next(iter(owners)))
            logits = net.predict_embeddings(query, net.prepare_support(*support))['logits'].float()
            if logits.shape != (len(ids), 2) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Complete finite gate logits required')
            scores = logits[:, 1] - logits[:, 0]
            labels = torch.tensor([ds.rows[i]['target'] for i in ids], device=scores.device).bool()
            p, u = scores[labels], scores[~labels]
            row = dict(case_id=case, observed_P=len(p), active_U=len(u),
                       query_record_ids=[ds.rows[i]['id'] for i in ids], rank_evaluable=bool(len(p) and len(u)))
            if row['rank_evaluable']:
                differences = p[:, None] - u[None, :]
                # Strict score separation: a tie cannot pass the training gate.
                values = torch.stack(((differences > 0).sum(), (differences == 0).sum(),
                                      (p.max() > u.max()).float(), p.max() - u.max())).cpu().tolist()
                wins, ties, hit, margin = values
                row.update(pairs=differences.numel(), wins=int(wins), ties=int(ties),
                           hit_at_1=hit, best_P_minus_best_U=margin)
                totals.append(row)
            row['seconds']=time.perf_counter()-start
            cases.append(row)
        if not totals:
            raise ValueError('No training P/U case can define a curriculum gate')
        pairs = sum(r['pairs'] for r in totals)
        metrics = dict(pair_win_rate=sum(r['wins'] for r in totals)/pairs,
                       pair_tie_rate=sum(r['ties'] for r in totals)/pairs,
                       hit_at_1=sum(r['hit_at_1'] for r in totals)/len(totals),
                       mean_margin=sum(r['best_P_minus_best_U'] for r in totals)/len(totals),
                       rank_evaluable_cases=len(totals), cases_without_P=sum(not r['observed_P'] for r in cases),
                       total_cases=len(cases), pairs=pairs, active_query_records=sum(len(v) for v in grouped.values()))
        return dict(metrics=metrics, cases=cases, query_GT_in_forward=False, validation_used_for_gate=False,
                    L0_reencoded=False, support='unchanged full eligible inner-train bank',
                    margin_definition='case mean of max(P score) minus max(active U score)',
                    hit_at_1_definition='case fraction with max(P score) strictly above max(active U score)',
                    scope='current cumulative TRAIN query set; not held-out quality evidence')
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(saved_rng)
