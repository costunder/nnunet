"""Read-only audit of the exact live P/U schedule and support-alignment mass.

This reports coefficient mass, not a replay of changing train-mode losses. It
does not select a new loss contract or apply the hypothetical group correction.
"""
from collections import Counter
import math

from l0_regions.donor_learning import LiveContext, groups, validate_rows


def audit_alignment_schedule(dataset, batch, seed, epoch):
    """Audit every scheduled tile without loading CT or executing the model.

    Per-positive integer bitsets prove Cartesian-pair coverage while avoiding
    storage of a Python tuple for every P x U pair. All observations/cases and
    patient groups are retained, including cases with only one observed class.
    """
    if type(batch) is not int or batch < 2:
        raise ValueError('Explicit physical batch >= 2 required')
    if seed is not None and type(seed) is not int:
        raise ValueError('Seed must be an integer or None')
    if type(epoch) is not int or epoch < 0:
        raise ValueError('Explicit nonnegative epoch field required')
    rows = dataset.rows
    by_case = validate_rows(rows)
    context = LiveContext(dataset, batch)
    order = list(groups(dataset, batch, seed, epoch))
    if len(order) != context.steps:
        raise ValueError('Seeded schedule differs in optimization-step count')
    uses = Counter(i for ids in order for i in ids)
    if uses != context.uses:
        raise ValueError('Seeded schedule changes observation multiplicity')
    if set(uses) != set(range(len(rows))):
        raise ValueError('Observation coverage incomplete')

    case_stats = {}
    negative_bits = {}
    seen_pairs = {}
    for case, ids in by_case.items():
        positive = [i for i in ids if rows[i]['target'] == 1]
        negative = [i for i in ids if rows[i]['target'] == 0]
        negative_bits[case] = {i: 1 << k for k, i in enumerate(negative)}
        seen_pairs[case] = dict.fromkeys(positive, 0)
        case_stats[case] = dict(
            case_id=case, patient_group=rows[ids[0]]['patient_group'],
            observations=len(ids), positive_count=len(positive),
            unobserved_count=len(negative), tiles=0, query_presentations=0,
            expected_ranking_pairs=len(positive) * len(negative),
            scheduled_ranking_pairs=0, repeated_ranking_pairs=0)

    group_tiles = Counter()
    group_episodes = Counter()
    previous_group = None
    for ids in order:
        case_names = {rows[i]['case_id'] for i in ids}
        group_names = {rows[i]['patient_group'] for i in ids}
        if len(case_names) != 1 or len(group_names) != 1:
            raise ValueError('Scheduled tile mixes recipient cases or groups')
        case, group = next(iter(case_names)), next(iter(group_names))
        if group != previous_group:
            group_episodes[group] += 1
            previous_group = group
        group_tiles[group] += 1
        stat = case_stats[case]
        stat['tiles'] += 1
        stat['query_presentations'] += len(ids)
        positive = [i for i in ids if rows[i]['target'] == 1]
        negative = [i for i in ids if rows[i]['target'] == 0]
        tile_bits = 0
        for i in negative:
            tile_bits |= negative_bits[case][i]
        stat['scheduled_ranking_pairs'] += len(positive) * len(negative)
        for i in positive:
            old = seen_pairs[case][i]
            stat['repeated_ranking_pairs'] += (old & tile_bits).bit_count()
            seen_pairs[case][i] = old | tile_bits

    total_unique_pairs = 0
    for case, stat in case_stats.items():
        unique = sum(bits.bit_count() for bits in seen_pairs[case].values())
        stat['unique_ranking_pairs'] = unique
        stat['missing_ranking_pairs'] = stat['expected_ranking_pairs'] - unique
        if stat['repeated_ranking_pairs'] or stat['missing_ranking_pairs']:
            raise ValueError(f'{case}: P x U ranking multiplicity/coverage failure')
        total_unique_pairs += unique
        stat['current_alignment_step_mean_weight'] = stat['tiles'] / context.steps
    if total_unique_pairs != context.pairs:
        raise ValueError('Global ranking-pair normalization differs from schedule')

    group_count = len(group_tiles)
    group_stats = []
    for group in sorted(group_tiles):
        cases = [s for s in case_stats.values() if s['patient_group'] == group]
        count = group_tiles[group]
        group_stats.append(dict(
            patient_group=group, case_ids=sorted(s['case_id'] for s in cases),
            observations=sum(s['observations'] for s in cases),
            positive_count=sum(s['positive_count'] for s in cases),
            unobserved_count=sum(s['unobserved_count'] for s in cases),
            tiles_per_patient_group=count,
            schedule_episode_count=group_episodes[group],
            current_alignment_step_mean_weight=count / context.steps,
            equal_group_step_mean_weight=1 / group_count,
            current_to_equal_group_weight_ratio=count * group_count / context.steps,
            hypothetical_group_balanced_per_tile_multiplier=context.steps / (group_count * count)))

    # Mirror production CE coefficients and average over all optimizer steps.
    # This verifies the actual presentation counts, not just an algebraic claim.
    class_mass = Counter()
    max_observation_error = 0.0
    for i, count in uses.items():
        target = rows[i]['target']
        coefficient = context.steps / (2 * context.counts[target] * context.uses[i])
        actual = count * coefficient / context.steps
        expected = 1 / (2 * context.counts[target])
        max_observation_error = max(max_observation_error, abs(actual - expected))
        class_mass[target] += actual
    balanced_ce = (max_observation_error <= 1e-14 and
                   all(math.isclose(class_mass[t], .5, rel_tol=0, abs_tol=1e-12)
                       for t in (0, 1)))
    if not balanced_ce:
        raise ValueError('Observation CE does not match global balanced mean')

    counts = list(group_tiles.values())
    ratios = [s['current_to_equal_group_weight_ratio'] for s in group_stats]
    return dict(
        schema='local_cnn_alignment_schedule_audit_v1', status='PASS',
        diagnostic_only=True, production_loss_changed=False,
        observed_scope='complete exact epoch schedule; no CT/model execution',
        epoch_field=epoch, seed=seed, physical_batch=batch,
        optimization_steps=context.steps, patient_group_count=group_count,
        case_count=len(by_case), unique_observations=len(rows),
        query_presentations=sum(uses.values()),
        alignment_contract=dict(
            confirmed_current='optimizer-step-weighted support alignment',
            intended_choice='UNRESOLVED: audit does not choose step-weighted or group-balanced',
            current_per_tile_multiplier=1.0,
            group_balanced_formula='S / (G * K_g)',
            hypothetical_correction_applied=False,
            interpretation='Weights are coefficient mass in an epoch step mean. '
                'Model/dropout and alignment losses may change between tiles; '
                'this is not a sum of identical frozen scalar losses.'),
        tile_count_distribution=dict(
            minimum=min(counts), maximum=max(counts), mean=context.steps / group_count,
            distinct_counts=sorted(set(counts)), all_groups_equal=len(set(counts)) == 1,
            minimum_current_to_equal_ratio=min(ratios),
            maximum_current_to_equal_ratio=max(ratios)),
        observation_presentation_distribution=dict(
            minimum=min(uses.values()), maximum=max(uses.values()),
            all_observations_present=True),
        actual_batch_size_counts={str(k): v for k, v in sorted(Counter(map(len, order)).items())},
        normalization=dict(
            ranking_pairs_expected=context.pairs,
            ranking_pairs_unique=total_unique_pairs,
            ranking_pairs_scheduled=sum(s['scheduled_ranking_pairs'] for s in case_stats.values()),
            ranking_pairs_repeated=0, ranking_pairs_missing=0,
            ranking_global_pair_mean_verified=True,
            observation_ce_global_balanced_mean_verified=balanced_ce,
            observation_ce_step_mean_class_mass={str(t): class_mass[t] for t in (0, 1)},
            maximum_observation_ce_weight_error=max_observation_error),
        patient_groups=group_stats,
        cases=[case_stats[k] for k in sorted(case_stats)])
