"""Bound target-signal diagnostics on frozen 128D L0/L1 representations.

Class separation uses labels only after the model forward. It is descriptive
within-case geometry, never held-out accuracy or evidence of CP efficacy.
The optional linear probe is a separate, explicitly requested DEBUG fit; it
does not train the original model or execute automatically on import.
"""
import math

import torch
from torch.nn import functional as F


FEATURE_DIMENSION = 128
FEATURE_SPACES = ('raw', 'l2_normalized')


def _bound_features(features, truth, rows=None):
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or features.shape[1] != FEATURE_DIMENSION or not len(features)
            or not features.is_floating_point()):
        raise ValueError('Nonempty original 128D floating-point stage features required')
    if (not isinstance(truth, torch.Tensor) or truth.shape != (len(features),)
            or truth.dtype != torch.long or truth.device != features.device
            or bool(((truth != 0) & (truth != 1)).any())):
        raise ValueError('Binary long truth aligned with every feature row on the same device required')
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError('Nonfinite stage features; no fallback')
    if rows is not None:
        if (not isinstance(rows, (list, tuple)) or len(rows) != len(features)
                or any(not isinstance(row, dict) or row.get('target') not in (0, 1) for row in rows)):
            raise ValueError('Complete original observation rows with binary targets required')
        expected = torch.tensor([row['target'] for row in rows], device=truth.device, dtype=torch.long)
        if not torch.equal(expected, truth):
            raise ValueError('Feature-row truth differs from original observation targets')
        if all('id' in row for row in rows) and len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Original feature-row observation identities must be unique')
    # FP64 accumulation resolves the small between-candidate variation in the
    # legacy path without changing the original representation or its graph.
    return features.detach().double(), truth.detach()


def _separation(values, truth):
    p, u = values[truth == 1], values[truth == 0]
    p_mean = p.mean(0) if len(p) else None
    u_mean = u.mean(0) if len(u) else None
    p_variance = (p-p_mean).square().sum(1).mean() if len(p) else None
    u_variance = (u-u_mean).square().sum(1).mean() if len(u) else None
    present = [value for value in (p_variance, u_variance) if value is not None]
    if present and not bool(torch.isfinite(torch.stack(present)).all()):
        raise FloatingPointError('Nonfinite within-class variance; no fallback')
    common = dict(within_observed_variance=float(p_variance) if p_variance is not None else None,
                  within_unobserved_variance=float(u_variance) if u_variance is not None else None,
                  variance_definition='mean squared Euclidean distance to the class mean; population trace')
    if p_mean is None or u_mean is None:
        return dict(class_mean_difference=None, class_mean_difference_squared=None,
                    within_variance_sum=None, fisher_ratio=None,
                    fisher_ratio_status='UNDEFINED_MISSING_OBSERVATION_CLASS', **common)
    difference = p_mean-u_mean
    numerator = difference.square().sum()
    denominator = p_variance+u_variance
    if not bool(torch.isfinite(torch.stack((numerator, denominator))).all()):
        raise FloatingPointError('Nonfinite class-separation reduction; no fallback')
    zero = bool(denominator == 0)
    ratio = None if zero else float(numerator/denominator)
    if ratio is not None and not math.isfinite(ratio):
        raise FloatingPointError('Nonfinite Fisher-like ratio; no fallback')
    return dict(class_mean_difference=difference.tolist(),
                class_mean_difference_squared=float(numerator),
                within_variance_sum=float(denominator), fisher_ratio=ratio,
                fisher_ratio_status='UNDEFINED_ZERO_WITHIN_CLASS_VARIANCE' if zero else 'MEASURED',
                **common)


def class_separation(features, truth, *, rows=None):
    """Describe raw and L2-normalized P/U geometry for one bound stage/case.

    ``rows`` should be the complete original ordered case observations when
    available; it checks truth against independent metadata. With no rows the
    tensor shape/binary binding is checked but independent binding is reported
    as unverified. Zero within-class variance yields None, never an epsilon-
    inflated Fisher ratio. A class containing one row has population variance 0.
    """
    values, truth = _bound_features(features, truth, rows)
    observed, unobserved = int((truth == 1).sum()), int((truth == 0).sum())
    return dict(status='MEASURED' if observed and unobserved else 'NOT_EVALUABLE',
                observed_count=observed, unobserved_count=unobserved,
                dimension=FEATURE_DIMENSION, candidates=len(values), finite=True,
                label_binding_verified=rows is not None,
                raw=_separation(values, truth),
                l2_normalized=_separation(F.normalize(values, dim=1), truth),
                zero_norm_candidates=int((values.norm(dim=1) == 0).sum()),
                scope='descriptive bound within-case target geometry; not held-out accuracy or CP efficacy',
                query_targets_used_only_after_forward=True)


def _cohort(cases, *, role, device=None):
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError('Explicit nonempty complete selected-case collection required')
    values, truths, indices, names, patients, observation_ids, lengths = [], [], [], [], [], [], []
    allowed = ('train', 'inner_train') if role == 'train' else ('validation', 'inner_val', 'heldout', 'test', 'inner_test')
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or case.get('split') not in allowed:
            raise ValueError(f'Explicit {role} split required for each frozen-feature case')
        rows = case.get('rows')
        if (not isinstance(rows, (list, tuple)) or not rows
                or any(not isinstance(row, dict) or any(key not in row for key in ('id', 'target', 'case_id', 'patient_group'))
                       for row in rows)):
            raise ValueError('Original row identities, case, patient group and truth required for linear probe')
        name, patient = case.get('case_id'), case.get('patient_group')
        if (not isinstance(name, str) or not name or not isinstance(patient, str) or not patient
                or any(row['case_id'] != name or row['patient_group'] != patient for row in rows)):
            raise ValueError('Frozen-feature case/patient identity differs from bound original rows')
        if case.get('all_case_candidates_retained') is not True:
            raise ValueError('Explicit retention of all original selected-case candidates required')
        x, y = _bound_features(case.get('features'), case.get('truth'), rows)
        if device is None:
            device = x.device
        if x.device != device:
            raise ValueError('All frozen train/held-out features must remain on the same explicit device')
        values.append(x)
        truths.append(y)
        indices.append(torch.full((len(x),), index, device=device, dtype=torch.long))
        names.append(name)
        patients.append(patient)
        observation_ids.extend(row['id'] for row in rows)
        lengths.append(len(x))
    if len(set(names)) != len(names) or len(set(observation_ids)) != len(observation_ids):
        raise ValueError('Duplicate selected case or observation identity in frozen-feature cohort')
    return dict(features=torch.cat(values), truth=torch.cat(truths), case_index=torch.cat(indices),
                case_ids=names, patient_groups=patients, observation_ids=observation_ids, lengths=lengths)


def _pair_indices(cohort):
    p = torch.where(cohort['truth'] == 1)[0]
    u = torch.where(cohort['truth'] == 0)[0]
    same_case = cohort['case_index'][p, None] == cohort['case_index'][None, u]
    positive, unobserved = torch.where(same_case)
    return p[positive], u[unobserved]


def _pair_report(scores, cohort, pairs):
    positive, unobserved = pairs
    margin = scores[positive]-scores[unobserved]

    def summarize(value):
        if not len(value):
            return dict(status='NOT_EVALUABLE', ranking_pairs=0, mean_pair_margin=None,
                        mean_pairwise_loss=None, pair_win_rate=None, exact_tie_rate=None)
        return dict(status='MEASURED', ranking_pairs=len(value), mean_pair_margin=float(value.mean()),
                    mean_pairwise_loss=float(F.softplus(-value).mean()),
                    pair_win_rate=float((value > 0).double().mean()), exact_tie_rate=float((value == 0).double().mean()))

    return dict(**summarize(margin), cases=[dict(case_id=name,
        candidates=length, **summarize(margin[cohort['case_index'][positive] == index]))
        for index, (name, length) in enumerate(zip(cohort['case_ids'], cohort['lengths']))])


def frozen_linear_pair_ranking_probe(train_cases, evaluation_cases, *, steps, lr, feature_space):
    """Explicit DEBUG 128D linear ranking fit, isolated from the source model.

    Every case dict requires case_id, patient_group, split, features, truth,
    original ordered rows, and all_case_candidates_retained=True. Train cases
    must declare train/inner_train; evaluation cases declare a held-out split.
    Case IDs, patient groups and observation IDs must be disjoint. A zero-
    initialized weight (no unidentifiable pairwise bias) is fitted with fresh
    AdamW, zero weight decay, on all within-case P x U pairs in one vectorized
    full-cohort objective. Evaluation labels never enter optimization.

    ``steps``, ``lr`` and raw/l2_normalized feature space are mandatory explicit
    probe settings. There is no hidden case/pair cap or production default.
    Full selected-case fitting is a diagnostic, not a production training run.
    """
    if type(steps) is not int or steps <= 0 or isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Explicit positive DEBUG steps and finite learning rate required')
    if feature_space not in FEATURE_SPACES:
        raise ValueError(f'Explicit frozen feature space required: {FEATURE_SPACES}')
    train = _cohort(train_cases, role='train')
    evaluation = _cohort(evaluation_cases, role='evaluation', device=train['features'].device)
    for key in ('case_ids', 'patient_groups', 'observation_ids'):
        if set(train[key]) & set(evaluation[key]):
            raise ValueError(f'Train/held-out {key} overlap; evaluation leakage is forbidden')
    if feature_space == 'l2_normalized':
        for cohort in (train, evaluation):
            cohort['features'] = F.normalize(cohort['features'], dim=1)
    train_pairs, evaluation_pairs = _pair_indices(train), _pair_indices(evaluation)
    if not len(train_pairs[0]):
        raise ValueError('Selected train cases contain no within-case observed/unobserved ranking pairs')
    weight = torch.zeros(FEATURE_DIMENSION, device=train['features'].device, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW([weight], lr=float(lr), weight_decay=0.)
    with torch.no_grad():
        before_train = _pair_report(train['features']@weight, train, train_pairs)
        before_evaluation = _pair_report(evaluation['features']@weight, evaluation, evaluation_pairs)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        score = train['features']@weight
        loss = F.softplus(score[train_pairs[1]]-score[train_pairs[0]]).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite explicit frozen linear-probe ranking loss')
        loss.backward()
        if weight.grad is None or not bool(torch.isfinite(weight.grad).all()):
            raise FloatingPointError('Missing/nonfinite frozen linear-head gradient')
        optimizer.step()
        if not bool(torch.isfinite(weight).all()):
            raise FloatingPointError('Nonfinite frozen linear-head optimizer update')
    with torch.no_grad():
        after_train = _pair_report(train['features']@weight, train, train_pairs)
        after_evaluation = _pair_report(evaluation['features']@weight, evaluation, evaluation_pairs)
    return dict(diagnostic_only=True, debug=True, feature_space=feature_space,
                feature_dimension=FEATURE_DIMENSION, trainable_parameters=FEATURE_DIMENSION,
                optimizer='fresh AdamW; zero initialization; no bias; weight_decay=0',
                steps=steps, learning_rate=float(lr), physical_batch=len(train['features']),
                gradient_accumulation_steps=1, effective_batch=len(train['features']),
                ranking_pairs=len(train_pairs[0]), all_selected_train_case_candidates_retained=True,
                frozen_feature_source_model_updated=False, checkpoint_written=False,
                training=dict(case_ids=train['case_ids'], candidates=len(train['features']), before=before_train, after=after_train),
                evaluation=dict(case_ids=evaluation['case_ids'], candidates=len(evaluation['features']), before=before_evaluation, after=after_evaluation),
                linear_weight=weight.detach().tolist(), parameter_delta_norm=float(weight.detach().norm()),
                device=str(weight.device), precision='FP64 diagnostic head and frozen-feature arithmetic',
                scope='explicit selected train-case frozen linear diagnostic; held-out selected cases only; not full accuracy or CP efficacy')
