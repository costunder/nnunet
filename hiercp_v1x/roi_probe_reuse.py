"""Read-only binding checks for previously measured source ROI costs."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('ROI reuse refuses symlinked evidence')
    return json.loads(path.read_text(encoding='utf-8'))


def verified_source_reuse(report_path, *, experiment, requests, config, snapshot,
                          medical_root,
                          spatial_sha256, failed_manifest_sha256, candidate_voxels,
                          rss_bytes, case_timeout_seconds):
    """Reuse only source PASS rows whose original requests/answers still match.

    Historical measurements remain historical: no fresh timing is fabricated.
    Failed rows are never adopted. Raw CT bytes are rehashed before reuse.
    """
    from .contracts import V1_ARCHIVE_SHA256
    graph = config.get('graph') if isinstance(config, dict) else None
    patch_size = graph.get('patch_size') if isinstance(graph, dict) else None
    if type(patch_size) is not int or patch_size <= 0:
        raise ValueError('Current ROI configuration requires an explicit positive integer graph.patch_size')
    path = Path(report_path).resolve(strict=True)
    report = read(path)
    if (report.get('format') != 'v1_roi_budget_probe_v1'
            or report.get('scope') != 'debug_geometry_only'
            or report.get('archived_source_sha256') != V1_ARCHIVE_SHA256
            or report.get('experiment') != str(Path(experiment).resolve())
            or report.get('failed_manifest_sha256') != failed_manifest_sha256
            or report.get('original_roi_max_voxels') != 8_000_000
            or report.get('candidate_roi_max_voxels') != candidate_voxels
            or report.get('originals_preserved') is not True
            or report.get('training_started') is not False
            or report.get('cache_publication_created') is not False):
        raise ValueError('Previous ROI report has a different experiment/input contract')
    budget = report.get('resource_budget', {})
    if (budget.get('rss_bytes') != rss_bytes
            or budget.get('case_timeout_seconds') != case_timeout_seconds):
        raise ValueError('Previous ROI measurement used different explicit resource limits')
    rows = report.get('measurements')
    identities = lambda seq: [(r['case_id'], r['sample_index']) for r in seq]
    if (not isinstance(rows, list) or len(rows) != len(requests)
            or report.get('failed_sample_requests') != len(requests)
            or identities(rows) != identities(requests)
            or len(set(identities(rows))) != len(rows)):
        raise ValueError('Previous ROI report has a different or duplicate request cohort')
    root = path.with_name(path.stem + '_requests')
    reused, files = {}, []
    raw_hashes = {}
    fields = ('case_id', 'sample_index', 'original_failure', 'geometry',
              'source_image_sha256', 'source_label_sha256')
    for ordinal, (row, current) in enumerate(zip(rows, requests)):
        if row.get('status') != 'PASS':
            continue
        if (row.get('geometry_equal') is not True
                or row.get('original_guard_reproduced') is not True
                or row.get('spatial_sha256') != spatial_sha256
                or row.get('requested_shape') != current['geometry']['requested_shape']
                or row.get('effective_shape') != current['geometry']['requested_shape']
                or row.get('roi_voxels') != current['geometry']['requested_voxels']
                or row.get('source_image_sha256') != current['source_image_sha256']
                or row.get('source_label_sha256') != current['source_label_sha256']
                or type(row.get('full_mask_voxels')) is not int or row['full_mask_voxels'] <= 0
                or type(row.get('source_component')) is not int or row['source_component'] < 1
                or row.get('model_input_shape') != [5]+[patch_size]*3
                or not row.get('limitation', '').startswith('Source ROI fields only;')):
            raise ValueError('Previous PASS is not an exact measured source ROI')
        for key in ('peak_rss_bytes', 'wall_seconds', 'geometry_seconds', 'load_seconds'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Previous ROI costs are missing or nonfinite')
        if row['peak_rss_bytes'] > rss_bytes or row['wall_seconds'] > case_timeout_seconds:
            raise ValueError('Previous ROI cost exceeds the current resource limits')
        request_path, answer_path = root/f'{ordinal:03d}_request.json', root/f'{ordinal:03d}_answer.json'
        request, answer = read(request_path), read(answer_path)
        if (request.get('config') != config
                or request.get('medical_root') != str(Path(medical_root).resolve())
                or request.get('snapshot') != str(Path(snapshot).resolve())
                or request.get('spatial_sha256') != spatial_sha256
                or request.get('candidate_roi_max_voxels') != candidate_voxels
                or request.get('shared') != str(Path(experiment).resolve()/'shared')
                or request.get('answer_path') != str(answer_path)
                or any(request.get('failure', {}).get(k) != current[k] for k in fields)
                or not isinstance(answer, dict) or not answer
                or any(row.get(k) != value for k, value in answer.items())):
            raise ValueError('Previous ROI request/answer differs from its claimed measurement')
        medical = Path(request['medical_root']).resolve()
        for name, suffix in (('image', f"image/{row['case_id']}_0000.nii.gz"),
                             ('label', f"labels/{row['case_id']}.nii.gz")):
            source_path = medical/'Data'/suffix
            if source_path not in raw_hashes:
                raw_hashes[source_path] = sha(source_path)
            if raw_hashes[source_path] != current[f'source_{name}_sha256']:
                raise ValueError('Previous measured raw CT bytes changed')
        result = dict(row, reused=True, reused_from_report=str(path),
                      reuse_report_sha256=sha(path),
                      reuse_request_sha256=sha(request_path), reuse_answer_sha256=sha(answer_path))
        reused[(row['case_id'], row['sample_index'])] = result
        files.extend([request_path, answer_path])
    return reused, dict(report=str(path), report_sha256=sha(path), records=len(reused),
        bound_sidecars={str(p): sha(p) for p in files},
        raw_sources={str(p): value for p, value in raw_hashes.items()})
