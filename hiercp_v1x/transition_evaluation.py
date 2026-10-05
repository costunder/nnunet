"""Whole native P/U evaluation; separate from the original eight-candidate task.

GT validates observation provenance and computes metrics. It is never passed to
the L0 or upper scorer. Production uses every inner-validation case and 128 U
per case. L0 can be chunked; the upper callback receives the complete case once.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

FORMAT = 'hiercp_transition_whole128_evaluation_v1'
SCORING_FORMAT = 'hiercp_transition_whole_case_scoring_v1'
QUERY_FIELDS = ('id', 'case_id', 'patient_group', 'center', 'donor_case_id',
                'donor_component', 'donor_group')


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _xyz(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3 or any(type(v) is not int or v < 0 for v in value):
        raise ValueError('Native integer center triplet required')
    return tuple(value)


def _cases(value, label):
    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v for v in value) or len(set(value)) != len(value):
        raise ValueError('Explicit unique case list required: ' + label)
    return value


def validate_cohort(inventory, *, case_ids=None, debug=False):
    """Validate signed-index metadata without decoding CT or changing any GT.

    The caller must bind the inventory bytes to its checkpoint/source receipt;
    this validator additionally checks the recorded P/U and donor provenance.
    """
    if type(debug) is not bool or not isinstance(inventory, Mapping):
        raise ValueError('Explicit metadata mapping and boolean DEBUG mode required')
    if (inventory.get('format') != 'native_local_cnn_inventory_v1'
            or inventory.get('complete') is not True or inventory.get('debug') is not debug
            or inventory.get('learning_policy') != 'same_donor_live_v1'):
        raise ValueError('Matching complete native same-donor inventory required')
    split = inventory.get('split', {})
    train = _cases(split.get('inner_train'), 'inner_train')
    val = _cases(split.get('inner_val'), 'inner_val')
    outer = _cases(split.get('outer_train'), 'outer_train')
    excluded = _cases(split.get('outer_val'), 'outer_val')
    if set(train) & set(val) or set(outer) != set(train) | set(val) or set(outer) & set(excluded):
        raise ValueError('Train/validation/outer exclusion contract differs')
    if not debug and (len(train), len(val), len(excluded)) != (84, 21, 26):
        raise ValueError('Production requires all 84/21 cases and untouched outer26')
    if case_ids is not None and (isinstance(case_ids, (str, bytes)) or not isinstance(case_ids, (list, tuple))):
        raise ValueError('Explicit case-id sequence required')
    selected = list(val) if case_ids is None else _cases(list(case_ids), 'evaluation')
    if not set(selected) <= set(val) or (not debug and set(selected) != set(val)):
        raise ValueError('Production evaluates all inner_val cases; subsets are DEBUG only')
    identities = inventory.get('identities', {}).get('cases', {})
    if not isinstance(identities, dict) or set(identities) != set(outer) | set(excluded):
        raise ValueError('Complete patient identity provenance required')
    groups = {}
    for name in (*outer, *excluded):
        group = identities[name].get('patient_group')
        if not isinstance(group, str) or not group:
            raise ValueError('Missing patient-group identity')
        partition = 'train' if name in train else 'val' if name in val else 'outer'
        if group in groups and groups[group] != partition:
            raise ValueError('Patient group crosses a held-out split')
        groups[group] = partition
    pool = inventory.get('donor_pool')
    if not isinstance(pool, list) or not pool:
        raise ValueError('Complete train-only donor pool required')
    donors = set()
    for donor in pool:
        key = (donor.get('case_id'), donor.get('component_id'))
        if key[0] not in train or type(key[1]) is not int or key[1] < 1 or key in donors:
            raise ValueError('Invalid or duplicate training donor provenance')
        donors.add(key)
    raw_rows = inventory.get('raw_records')
    if not isinstance(raw_rows, list) or any(not isinstance(row, dict) for row in raw_rows):
        raise ValueError('Raw observation provenance required')
    raw = {row.get('case_id'): row for row in raw_rows}
    if len(raw) != len(raw_rows) or not set(outer) <= set(raw):
        raise ValueError('Missing or duplicated raw-case provenance')
    rows = inventory.get('records')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Complete observation rows required')
    by_case = {name: [] for name in outer}
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or row.get('case_id') not in by_case:
            raise ValueError('Observation outside retained outer_train cohort')
        name = row['case_id']
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids:
            raise ValueError('Unique observation identity required')
        ids.add(row['id'])
        if type(row.get('target')) is not int or row['target'] not in (0, 1):
            raise ValueError('P/U targets must be literal integer zero or one')
        if type(row.get('donor_component')) is not int or row['donor_component'] < 1:
            raise ValueError('Literal eligible donor component identity required')
        if row['target'] == 1 and (type(row.get('component')) is not int or row['component'] < 1):
            raise ValueError('Observed P requires its literal raw component identity')
        _xyz(row.get('center'))
        if (row.get('patient_group') != identities[name]['patient_group']
                or (row.get('donor_case_id'), row.get('donor_component')) not in donors
                or row.get('donor_group') != identities[row['donor_case_id']]['patient_group']
                or row['donor_group'] == row['patient_group']):
            raise ValueError('Independent train-only donor/patient binding differs')
        by_case[name].append(dict(row))
    from tools.v22_candidate_order import record_key
    for name, case in by_case.items():
        info = raw[name]
        for key in ('image_sha256', 'label_sha256'):
            digest = info.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Raw CT/annotation SHA256 provenance required')
        positives = info.get('positives')
        centers = info.get('comparison', {}).get('centers')
        if not isinstance(positives, list) or not isinstance(centers, list) or len(centers) != 128:
            raise ValueError('Every retained case requires all recorded P and 128 U')
        if any(not isinstance(p, dict) or type(p.get('component')) is not int or p['component'] < 1 for p in positives):
            raise ValueError('Actual raw positive component/anchor provenance required')
        expected_p = Counter((p['component'], _xyz(p.get('center'))) for p in positives)
        actual_p = Counter((r.get('component'), _xyz(r['center'])) for r in case if r['target'] == 1)
        expected_u = Counter(_xyz(c) for c in centers)
        actual_u = Counter(_xyz(r['center']) for r in case if r['target'] == 0)
        if (actual_p != expected_p or actual_u != expected_u or any(n != 1 for n in expected_u.values())
                or set(expected_u) & {xyz for _, xyz in expected_p}
                or any(r.get('component') is not None for r in case if r['target'] == 0)):
            raise ValueError('Whole P/U observation provenance/coverage differs')
        if len({(r['donor_case_id'], r['donor_component'], r['donor_group']) for r in case}) != 1:
            raise ValueError('All case observations must use the same independent donor')
        keys = [record_key(row) for row in case]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate geometry identity within a complete case')
    raw_positive_donors = {(name, p['component']) for name in train for p in raw[name]['positives']}
    if not donors <= raw_positive_donors:
        raise ValueError('Donor pool component missing from actual training annotation provenance')
    selected_rows = [r for name in selected for r in by_case[name]]
    native_assignment = validate_native_assignments(inventory)
    return dict(case_ids=selected, rows=selected_rows, by_case={name: by_case[name] for name in selected},
                debug=debug, production_full_inner_val=not debug,
                inventory_metadata_sha256=_sha(inventory), cohort_sha256=_sha(selected_rows),
                native_assignment=native_assignment,
                records=len(selected_rows), observed_P=sum(r['target'] for r in selected_rows),
                unobserved_U=sum(1-r['target'] for r in selected_rows))


def validate_native_assignments(inventory):
    """Compare every assigned donor with the actual native seed-fixed algorithm."""
    if inventory.get('config', {}).get('seed') != 42:
        raise ValueError('Bound native seed42 configuration required')
    # Use the identical signed assignment function without importing the
    # current preparation/model stack into a historical model namespace.
    from .native30_data_contract import assignment
    expected = {row['id']: row for row in assignment(inventory, 42)}
    actual = {row['id']: row for row in inventory['records']}
    if set(actual) != set(expected) or any(
        actual[identity].get(key) != value for identity, row in expected.items() for key, value in row.items()
    ):
        raise ValueError('Native signed seed-fixed donor assignment differs')
    return dict(policy='same_donor_live_v1', seed=42, records=len(actual),
                assignment_sha256=_sha([expected[key] for key in sorted(expected)]))


def _scoring_contract(value, ids, upper_invocations):
    if not isinstance(value, Mapping) or value.get('format') != SCORING_FORMAT:
        raise ValueError('Explicit whole-case scoring execution contract required')
    if (value.get('l0_only_chunking') is not True or value.get('upper_chunking') is not False
            or value.get('upper_execution') != 'single_joint_case'
            or value.get('query_GT_in_forward') is not False
            or value.get('scored_record_ids') != ids
            or type(value.get('upper_invocations')) is not int
            or value['upper_invocations'] != upper_invocations):
        raise ValueError('Independent upper chunks or GT-bearing scoring cannot claim whole-case evaluation')
    return dict(value)


def evaluate_scores(inventory, scores_by_id, *, case_ids=None, debug=False, scoring_contract=None):
    """Evaluate one finite score for each actual held-out native observation."""
    cohort = validate_cohort(inventory, case_ids=case_ids, debug=debug)
    rows = cohort['rows']; ids = [r['id'] for r in rows]
    if not isinstance(scores_by_id, Mapping) or set(scores_by_id) != set(ids):
        raise ValueError('One score for every observation; missing/extra scores forbidden')
    scores = []
    for identity in ids:
        score = scores_by_id[identity]
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError('Every actual score must be finite numeric metadata')
        scores.append(float(score))
    execution = None if scoring_contract is None else _scoring_contract(scoring_contract, ids, len(cohort['case_ids']))
    from tools.v22_candidate_order import record_key, candidate_order, TIE_POLICY
    from tools.v22_rank_objective import ranking_metrics
    keys = [record_key(row) for row in rows]
    # Reuse native metric definitions verbatim rather than reinterpret R@1 as top1.
    native, reports = ranking_metrics(scores, [r['target'] for r in rows],
                                     [r['case_id'] for r in rows], candidate_keys=keys)
    if not math.isfinite(native['ranking_pairwise_loss']):
        raise ValueError('Native FP32 reference metric overflowed; no substituted result')
    score_map = dict(zip(ids, scores)); cases = []; wins = ties = pair_count = 0; losses = []
    for report in reports:
        case = cohort['by_case'][report['case_id']]
        values = [score_map[r['id']] for r in case]; mean = math.fsum(values)/len(values)
        p = [score_map[r['id']] for r in case if r['target'] == 1]
        u = [score_map[r['id']] for r in case if r['target'] == 0]
        for positive in p:
            for negative in u:
                difference = positive-negative
                wins += difference > 0; ties += difference == 0; pair_count += 1
                losses.append(max(-difference, 0.) + math.log1p(math.exp(-abs(difference))))
        order = candidate_order(values, [record_key(row) for row in case]).tolist()
        cases.append(dict(report, observed_P=len(p), unobserved_U=len(u), records=len(case),
                          score_std=math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values)),
                          score_min=min(values), score_max=max(values),
                          scored_record_ids=[r['id'] for r in case],
                          ordered_record_ids=[case[i]['id'] for i in order],
                          case_scores=[dict(record_id=r['id'], candidate_key=record_key(r),
                                            score=score_map[r['id']], observed=r['target']) for r in case]))
    valid = [r for r in reports if r['rank_evaluable']]
    if pair_count != native['ranking_pairs']:
        raise ValueError('Native metric pair denominator differs from full-case P times U')
    mean = math.fsum(scores)/len(scores)
    metrics = dict(case_first_P_mrr=native['ranking_mrr'],
                   case_hit_at_1=sum(r['first_observed_rank'] == 1 for r in valid)/len(valid),
                   **{f'observed_micro_recall_at_{k}': native[f'ranking_recall_at_{k}'] for k in (1,5,10)},
                   P_U_pair_win_rate=wins/pair_count, P_U_pair_tie_rate=ties/pair_count,
                   P_U_softplus_loss=math.fsum(losses)/pair_count,
                   score_std=math.sqrt(math.fsum((v-mean)**2 for v in scores)/len(scores)),
                   score_min=min(scores), score_max=max(scores))
    return dict(format=FORMAT, task='native_observed_P_vs_unobserved_U',
                GT_is_donor_compatibility=False, original_eight_candidate_metrics=False,
                debug=debug, full_128_U_per_case=True, production_full_inner_val=not debug,
                quality_verified=False, full_training_complete=False,
                cohort={k:v for k,v in cohort.items() if k not in ('rows','by_case')},
                denominators=dict(cases=len(cases), rank_evaluable_cases=len(valid),
                                  zero_P_cases=len(cases)-len(valid), observed_P=cohort['observed_P'],
                                  unobserved_U=cohort['unobserved_U'], P_U_pairs=pair_count,
                                  case_hit_at_1=len(valid), observed_micro_recall=cohort['observed_P']),
                metrics=metrics, native_metrics=native, cases=cases, tie_policy=TIE_POLICY,
                pair_loss_precision='stable float64 softplus; native_metrics preserves native FP32 reference',
                scores_sha256=_sha(dict(zip(ids, scores))), scoring_contract=execution,
                execution_contract_bound=execution is not None,
                scoring_callbacks_executed=False, callback_internal_execution_verified=False,
                raw_CT_execution_verified=False)


@dataclass(frozen=True)
class PreparedL0Batch:
    inputs: object
    record_ids: tuple[str, ...]
    query_GT_in_inputs: bool


def run_scoring(inventory, batch_provider, encode_l0, score_case_joint, *, l0_batch_size,
                case_ids=None, debug=False, output=None):
    """Chunk only L0, then invoke the complete-case upper callback exactly once.

    provider(query_rows)->PreparedL0Batch; encode_l0(inputs)->[N,D] Tensor;
    score_case_joint(all_embeddings, query_rows)->{'scores': [N] Tensor,
    'contract': whole-case scoring contract}. Query rows omit target/component.
    The caller must set the neural model to eval and manage RNG/mode restoration.
    The callback's internal upper implementation still needs source/smoke audit.
    """
    if type(l0_batch_size) is not int or l0_batch_size < 1:
        raise ValueError('Explicit positive measured physical L0 batch required')
    cohort = validate_cohort(inventory, case_ids=case_ids, debug=debug)
    import torch
    score_map = {}; all_ids = []; case_execution = []
    with torch.no_grad():
        from tqdm import tqdm
        for name, rows in tqdm(cohort['by_case'].items(),total=len(cohort['by_case']),desc='whole P+128U evaluation',unit='case'):
            query = tuple({key:list(r[key]) if key == 'center' else r[key] for key in QUERY_FIELDS} for r in rows)
            pieces = []
            for start in range(0, len(query), l0_batch_size):
                chunk = query[start:start+l0_batch_size]
                batch = batch_provider(chunk)
                ids = tuple(r['id'] for r in chunk)
                if (not isinstance(batch, PreparedL0Batch) or batch.record_ids != ids
                        or batch.query_GT_in_inputs is not False):
                    raise ValueError('L0 input identity or GT exclusion receipt differs')
                features = encode_l0(batch.inputs)
                if (not isinstance(features, torch.Tensor) or features.ndim != 2
                        or features.shape[0] != len(chunk) or not features.is_floating_point()
                        or not bool(torch.isfinite(features).all())):
                    raise ValueError('Actual finite batched L0 embeddings required')
                pieces.append(features)
            joint = torch.cat(pieces, dim=0)
            result = score_case_joint(joint, query)
            ids = [r['id'] for r in rows]
            if not isinstance(result, Mapping):
                raise ValueError('Actual joint upper result mapping required')
            contract = _scoring_contract(result.get('contract'), ids, 1)
            values = result.get('scores')
            if (not isinstance(values, torch.Tensor) or values.shape != (len(rows),)
                    or not values.is_floating_point() or not bool(torch.isfinite(values).all())):
                raise ValueError('One finite real upper score per complete-case observation required')
            score_map.update(zip(ids, values.detach().to('cpu', torch.float64).tolist()))
            all_ids.extend(ids)
            case_execution.append(dict(case_id=name, records=len(rows), l0_chunks=len(pieces),
                                       upper_invocations=1, callback_contract=contract))
            del result, values, joint, pieces, features, batch
    contract = dict(format=SCORING_FORMAT, scored_record_ids=all_ids, l0_only_chunking=True,
                    upper_chunking=False, upper_execution='single_joint_case', query_GT_in_forward=False,
                    upper_invocations=len(case_execution), l0_batch_size=l0_batch_size, cases=case_execution,
                    verification_scope='wrapper enforces complete-case callback; callback source and real-CT smoke audit required')
    if _sha(inventory) != cohort['inventory_metadata_sha256']:
        raise ValueError('Scoring callbacks changed the bound observation inventory')
    report = evaluate_scores(inventory, score_map, case_ids=case_ids, debug=debug, scoring_contract=contract)
    report['scoring_callbacks_executed'] = True
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    return report
