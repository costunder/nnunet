"""Read-only source audit; writes new evidence, never restores a model or trains."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    out = ROOT / 'work/v222_graph_history_audit_20260923'
    out.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    selections = {
        'versions/v2.2/history/relations/before-observed': [
            'hiercp_v221/geometry.py', 'hiercp_v221/spatial.py',
            'hiercp_v221/sample.py', 'hiercp_v221/data.py',
            'config/prompt_graph_v221.json'],
        'versions/v2.2/history/observed/before-clusters': [
            'hiercp_v222/inputs.py', 'docs/pipeline_v222.md'],
    }
    for directory, paths in selections.items():
        folder = ROOT / directory
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        with zipfile.ZipFile(folder / 'source.zip') as archive:
            for name in paths:
                data = archive.read(name)
                actual = digest(data)
                assert actual == manifest['files'][name], (directory, name)
                snapshots[directory + '/' + name] = {
                    'archive_sha256': actual, 'manifest_verified': True,
                    'current_file_equal': (ROOT / name).read_bytes() == data,
                }
    receipt_path = ROOT / 'work/v22_no_interior_r5_20260922/debug1/verification.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    # Historical real-CT evidence is re-read and bound to source hashes, not rerun.
    matching_maps = [v for v in receipt.values() if isinstance(v, dict)
                     and 'hiercp_v22/geometry.py' in v]
    assert len(matching_maps) == 1
    identities = matching_maps[0]
    verified = {}
    for name in ('geometry', 'spatial', 'sample', 'local', 'schema'):
        relative = f'hiercp_v22/{name}.py'
        actual = digest((ROOT / relative).read_bytes())
        assert actual == identities[relative], relative
        other = ROOT / f'hiercp_v221/{name}.py'
        assert other.read_bytes() == (ROOT / relative).read_bytes(), name
        verified[relative] = actual
    cfg = json.loads((ROOT / 'config/prompt_graph_v221.json').read_text(encoding='utf-8'))
    base = json.loads((ROOT / 'config/train.json').read_text(encoding='utf-8'))
    from hiercp_v222.contracts import source_identity, verify_v1
    init = json.loads((ROOT / 'work/v222_raw_ct_r3_training_20260923/gnn_vram_affine/initialization.json').read_text(encoding='utf-8'))
    assert source_identity() == init['source_identity']
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'historical source/receipt audit; no model restoration or training',
        'snapshots': snapshots,
        'historical_real_ct_receipt': receipt_path.relative_to(ROOT).as_posix(),
        'historical_receipt_sha256': digest(receipt_path.read_bytes()),
        'historical_real_ct_rerun': False,
        'historical_graphs': receipt['graphs'],
        'historical_source_hashes_verified': verified,
        'historical_limitations': {k: receipt[k] for k in (
            'full_training', 'full_evaluation', 'full_training_objective_complete', 'limitation')},
        'v221_status': {k: cfg[k] for k in (
            'version_status', 'implementation_scope', 'l1_contract_status',
            'training_objective_status', 'loss_weights')},
        'context_sampling': {k: base['graph'][k] for k in ('sample_context_nodes', 'sample_hops')},
        'current_source_matches_stopped_run': True,
        'frozen_v1_verification': verify_v1(),
        'model_changed': False, 'training_started': False,
    }
    target = out / 'audit.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'evidence': str(target), 'archive_files_verified': len(snapshots),
                      'historical_graphs': receipt['graphs'], 'current_model_unchanged': True}))


if __name__ == '__main__':
    main()
