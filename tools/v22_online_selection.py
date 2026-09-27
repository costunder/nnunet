"""Native event contract: retain raw scores, consume the verified eligible choice."""
import numpy as np
from tools.v22_candidate_order import candidate_key, candidate_order

CONTRACT = 'observed_rank_native_selection_v1'


def validate_selection(value):
    if value.get('selection_contract') != CONTRACT:
        raise ValueError('Explicit observed ranking selection required')
    centers = np.asarray(value['centers'])
    scores = np.asarray(value['scores'], dtype=np.float64)
    eligible = np.asarray(value['eligible'])
    n = len(scores)
    if (scores.ndim != 1 or not np.isfinite(scores).all() or centers.shape != (n, 3)
            or not np.issubdtype(centers.dtype, np.integer) or eligible.shape != (n,)
            or eligible.dtype != bool or len(np.unique(centers, axis=0)) != n):
        raise ValueError('Invalid complete candidate scores/geometry/eligibility')
    keys = [candidate_key(value['recipient'], value['donor'], value['component'], c) for c in centers]
    order = candidate_order(scores, keys)
    valid = [int(i) for i in order if eligible[i]]
    expected = valid[0] if valid else None
    selected = value['selected_index']
    if selected is not None and type(selected) is not int:
        raise ValueError('Integer selected index or None required')
    if selected != expected or value.get('keep_original') is not (expected is None):
        raise ValueError('Selection is not the eligible highest score with the recorded tie policy')
    if value.get('score_override') is not False or value.get('filter_after_model_scoring') is not True:
        raise ValueError('Scores must remain unchanged before eligibility filtering')
    return selected


def selection_from_report(report, placements):
    if not placements:
        raise ValueError('Native event requires its explicit proposal pool')
    first = placements[0]
    rows = sorted(report['ranked_candidates'], key=lambda r: r['index'])
    if [r['index'] for r in rows] != list(range(len(placements))):
        raise ValueError('Incomplete scored candidate report')
    for row, placement in zip(rows, placements):
        if row['center'] != list(placement.center):
            raise ValueError('Ranked and placed candidate centers differ')
    value = dict(selection_contract=CONTRACT, recipient=first.recipient, donor=first.donor,
        component=first.component, centers=[list(p.center) for p in placements],
        scores=[r['model_score'] for r in rows], eligible=[r['eligible'] for r in rows],
        selected_index=report['selected_index'], keep_original=report['keep_original'],
        score_override=report['score_override'], filter_after_model_scoring=report['filter_after_model_scoring'],
        ranked_candidates=report['ranked_candidates'])
    validate_selection(value)
    return value


class RankedEventLoaderMixin:
    """Same five draws as the native baseline; no second argmax or donor redraw."""
    def _sample_paste_plan(self, case_id):
        from custom_trainers.nnUNetTrainer_OnlinePairedCP import TRAINER_FORMAT, _stable_u64
        names = self._source_entry_names(case_id)
        rng = self._rng()
        apply = float(rng.random()) < self.online_bank.cp_probability
        source_u, candidate_u, scale_u, shift_u = [float(rng.random()) for _ in range(4)]
        index = min(len(names)-1, int(np.floor(source_u*len(names)))) if names else -1
        name = names[index] if names else ''
        token = _stable_u64(TRAINER_FORMAT, self.online_epoch, str(case_id), int(apply and bool(name)),
            source_u.hex(), candidate_u.hex(), scale_u.hex(), shift_u.hex())
        if not apply or not name:
            return None, token
        entry = self._load_selected_source(case_id, index)
        selected = self.online_bank.selected_index(entry)
        if selected is None:
            return None, token
        lo, hi = self.online_bank.intensity_scale
        low, high = self.online_bank.intensity_shift_hu
        return self._make_paste_plan(entry, selected, lo+scale_u*(hi-lo), low+shift_u*(high-low), case_id), token
