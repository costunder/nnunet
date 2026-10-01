"""Format saved causal diagnostic JSON using only the standard library."""


def format_causal_summary(comparison, *, mode_ranking=False):
    """Compact complete causal signals; reads measured JSON only, no pager."""
    if comparison.get('causal_probe') is not True:
        raise ValueError('Explicit measured causal comparison required')
    def field(value, *keys):
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value
    def number(value):
        return 'UNAVAILABLE' if value is None else f'{value:.12g}'
    def text(value):
        return 'UNAVAILABLE' if value is None else str(value)
    def score_fields(score):
        win, tie = field(score, 'pair_win_rate'), field(score, 'exact_tie_rate')
        half_tie = None if win is None or tie is None else win + 0.5*tie
        return ('score std/min/max='+'/'.join(number(field(score, key))
                for key in ('score_std', 'score_min', 'score_max'))
            +' | exact-tie='+number(tie)+' strict-pair-win='+number(win)
            +' half-tie-pair-win(computed)='+number(half_tie)
            +' mean-pair-loss='+number(field(score, 'mean_pairwise_loss')))
    def recall_fields(case, metrics):
        ranks = case.get('observed_ranks')
        complete = (case.get('all_case_candidates_retained') is True
            and isinstance(ranks, list)
            and all(type(rank) is int and rank >= 1 for rank in ranks))
        positives = len(ranks) if complete else None
        records = case.get('records')
        unknown = (records-positives if positives is not None and type(records) is int
            and records >= positives else None)
        recalls = []
        for k in (1, 5, 10):
            hits = sum(rank <= k for rank in ranks) if complete else None
            maximum = min(k, positives)/positives if positives else None
            recalls.append(f'recall@{k}='+number(field(metrics, f'ranking_recall_at_{k}'))
                +' hits(computed)='+text(hits)+'/'+text(positives)
                +' max(computed)='+number(maximum))
        return positives, unknown, ' | '.join(recalls)
    lines=['CAUSAL DEBUG | fitted-case evaluation / exact-forward loss and Adam / BN controls',
        'Cloned short updates only. Candidate spread and gradient size are not accuracy.',
        'Tile pre/post: same saved episodic support + frozen teacher, eval BN/dropout off.',
        'Whole fitted case: full eligible saved support + newly fitted own teacher, eval mode.',
        'First-positive MRR is reciprocal first-positive rank, averaged over evaluable cases; recall@k covers all eligible positives.',
        'Eligible P / unknown U, recall hits/max and half-tie pair-win are computed from saved fields; missing fields are UNAVAILABLE.',
        'Full-descent cosine compares ranking gradient with minus full Adam delta; delta rank-full compares optimizer deltas.']
    for branch in comparison.get('branches', []):
        lines.append('BRANCH '+text(branch.get('branch'))+' | actual fitted cases='+text(branch.get('fitted_case_ids')))
        if 'fitted_case_timeline' not in branch:
            lines.append('  fitted-case timeline=UNAVAILABLE')
        for point in branch.get('fitted_case_timeline', []):
            for group in ('train','validation'):
                for case in (field(point, 'evaluation', group, 'cases') or []):
                    m,s=case.get('metrics', {}),case.get('score', {})
                    positives, unknown, recalls = recall_fields(case, m)
                    lines.append('  fitted '+text(case.get('case_id'))+' after='+text(point.get('after_updates'))
                        +' N='+text(case.get('records'))+' eligibleP(computed)='+text(positives)
                        +' unknownU(computed)='+text(unknown)
                        +' | margin='+number(field(s, 'mean_positive_minus_unobserved'))
                        +' first-positive MRR='+number(field(m, 'ranking_mrr')))
                    lines.append('    '+score_fields(s))
                    lines.append('    '+recalls)
                    stages = field(case, 'trace', 'stages')
                    lines.append('    target Fisher raw/normalized | '+(' | '.join(
                        text(stage.get('stage'))+'='+number(field(stage, 'target_signal', 'raw', 'fisher_ratio'))+'/'
                        +number(field(stage, 'target_signal', 'l2_normalized', 'fisher_ratio'))
                        for stage in stages) if stages is not None else 'UNAVAILABLE'))
        for step in branch.get('updates', []):
            causal=step.get('causal', {});a=field(causal, 'tile_before', 'score');b=field(causal, 'tile_after', 'score')
            lines.append('  step='+text(step.get('step'))+' schedule='+text(step.get('schedule_index'))
                +' case='+text(step.get('case_id'))+' P/U='+text(step.get('observed'))+'/'+text(step.get('unobserved'))
                +' | frozen tile margin '+number(field(a, 'mean_positive_minus_unobserved'))
                +'->'+number(field(b, 'mean_positive_minus_unobserved'))
                +' win '+number(field(a, 'pair_win_rate'))+'->'+number(field(b, 'pair_win_rate'))
                +' | exact full-Adam parity='+text(causal.get('shadow_full_matches_actual_update')))
            lines.append('    frozen tile before | '+score_fields(a))
            lines.append('    frozen tile after | '+score_fields(b))
            finite = causal.get('finite_shadow_scores')
            if finite:
                lines.append('    finite shadow | fixed pre-forward BN/dropout off; native CNN re-encoded; same frozen teacher; train-mode gradients')
                for name in ('no_change', 'rank_only', 'full'):
                    arm = field(finite, 'arms', name)
                    score = field(arm, 'score')
                    lines.append('      '+name+' | margin='+number(field(score, 'mean_positive_minus_unobserved'))
                        +' | '+score_fields(score)+' | scoring seconds='+number(field(arm, 'synchronized_scoring_seconds'))
                        +' common score change='+number(field(arm, 'score_change_from_no_change', 'mean'))
                        +' P-minus-U change='+number(field(arm, 'score_change_from_no_change', 'positive_minus_unobserved_mean_change')))
            objective=causal.get('objective_direction', {})
            for module in ('CNN','readout_fusion','L1','L2'):
                row=field(objective, 'module_gradients', module);d=field(objective, 'delta_cosines', module)
                lines.append('    '+module+' | grad norms rank/CE/align/full='+'/'.join(
                    number(field(row, 'terms', key, 'norm')) for key in ('ranking','observation_ce','alignment','full'))
                    +' | cos rank-CE/align/full='+'/'.join(number(field(row, 'cosines', 'ranking_vs_'+key, 'cosine'))
                        for key in ('observation_ce','alignment','full'))
                    +' | cos delta rank-full='+number(field(d, 'rank_only_vs_full', 'cosine'))
                    +' | cos ranking-gradient/full-descent='+number(field(d, 'ranking_gradient_vs_full_descent', 'cosine')))
            mode_control=causal.get('mode_control')
            if mode_control:
                for mode in mode_control.get('modes', []):
                    score=mode.get('score', {})
                    ranking = field(mode, 'per_loss_query_CNN_gradients', 'ranking')
                    ranking_fields = (' weighted-rank-loss='+number(field(ranking, 'weighted_loss'))
                        +' rank-query-grad='+number(field(ranking, 'query_embedding_gradient_norm'))
                        +' rank-CNN-grad='+number(field(ranking, 'CNN_parameter_gradient_norm'))) if mode_ranking else ''
                    lines.append('    BN/dropout '+text(mode.get('mode'))+' | margin='+number(field(score, 'mean_ranking_margin'))
                        +' win='+number(field(score, 'pair_win_rate'))+' weighted-align='+number(mode.get('weighted_alignment_loss'))
                        +' align-query-grad='+number(field(mode, 'per_loss_query_CNN_gradients', 'alignment', 'query_embedding_gradient_norm'))
                        +ranking_fields)
                control=mode_control.get('frozen_BN_dropout_off_full_update', {})
                a,b=field(control, 'before', 'score'),field(control, 'after', 'score')
                lines.append('    frozen-BN cloned full update | margin='
                    +number(field(a, 'mean_ranking_margin'))+'->'+number(field(b, 'mean_ranking_margin'))
                    +' win='+number(field(a, 'pair_win_rate'))+'->'+number(field(b, 'pair_win_rate'))
                    +' | fresh AdamW, original clipping')
    return '\n'.join(lines)
