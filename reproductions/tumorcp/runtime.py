"""Build an auditable, isolated paper-profile runtime from preserved TumorCP code.

The upstream checkout is never modified. This module needs only Python's standard
library, so source preparation does not import PyTorch or allocate a GPU.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil


CP_MODE_SOURCE = '''def _paper_tumorcp_mode(gate_draw, mode_draw, probability):
    """One event at most: 80% CP, then equal intra/inter probability."""
    if not (0 <= gate_draw < 1 and 0 <= mode_draw < 1):
        raise ValueError("TumorCP draws must be in [0, 1)")
    if not 0 <= probability <= 1:
        raise ValueError("TumorCP probability must be in [0, 1]")
    if gate_draw >= probability:
        return None
    return "intra" if mode_draw < 0.5 else "inter"
'''

PACK_TUMORS_SOURCE = '''def _paper_pack_tumors(tumors):
    """Store variable-size arrays without converting their voxel dtype to object."""
    packed = np.empty((len(tumors), 2), dtype=object)
    for index, (data, mask) in enumerate(tumors):
        packed[index, 0] = data
        packed[index, 1] = mask
    return packed
'''


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one {label} source block; found {count}")
    return text.replace(old, new, 1)


def _patch_paths(text: str) -> str:
    start = text.index("base = '/dfs/data/MyTasks/TumorCP/data/'")
    end = text.index("if base is not None:", start)
    replacement = '''# Isolated paper-reproduction paths; never write into another experiment.
base = os.environ.get("nnUNet_raw_data_base")
preprocessing_output_dir = os.environ.get("nnUNet_preprocessed")
network_training_output_dir_base = os.environ.get("RESULTS_FOLDER")
if not all((base, preprocessing_output_dir, network_training_output_dir_base)):
    raise RuntimeError("Set nnUNet_raw_data_base, nnUNet_preprocessed, RESULTS_FOLDER for the isolated KiTS19 reproduction")

'''
    return text[:start] + replacement + text[end:]


def _patch_loader(text: str) -> str:
    start = text.index("            if self.cp_configs['do_cp'] and np.random.uniform() <= self.cp_configs['p_cp']:")
    end = text.index("            # print('############ after'", start)
    replacement = '''            # Paper profile: exactly one CP gate, then intra/inter 50:50.
            if self.cp_configs.get('paper_profile', False):
                mode = _paper_tumorcp_mode(float(np.random.uniform()),
                                           float(np.random.uniform()),
                                           self.cp_configs['p_cp'])
                event = dict(target_case=str(i), donor_case=None, mode=mode,
                             attempted=mode is not None, status='gate_off', paste_voxels=0)
                if mode == 'intra' and i in self._paper_empty_tumor_ids:
                    event['status'] = 'no_source'
                elif mode == 'intra':
                    tumors = np.load(self._paper_tumor_files[i], allow_pickle=True)
                    event['donor_case'] = str(i)
                elif mode == 'inter':
                    donor_keys = [key for key in self._paper_tumor_files if key != i]
                    if not donor_keys:
                        raise RuntimeError("Inter-CP requires another tumor-bearing training patient")
                    tumor_key = np.random.choice(donor_keys)
                    tumors = np.load(self._paper_tumor_files[tumor_key], allow_pickle=True)
                    event['donor_case'] = str(tumor_key)
                if event['donor_case'] is not None:
                    self._paper_last_paste_voxels = 0
                    case_all_data = self.aug_one_pair(source_tumors=tumors,
                                                      target_data=case_all_data,
                                                      target_organ_positions=properties['class_locations'][1])
                    event['paste_voxels'] = self._paper_last_paste_voxels
                    event['status'] = 'applied' if self._paper_last_paste_voxels else 'empty_transform'
                paper_cp_audit.append(event)
            else:
'''
    original = text[start:end]
    # Keep the official unmodified profile available; only the paper trainer
    # activates the alternative event schedule above.
    replacement += ''.join('    ' + line if line.strip() else line for line in original.splitlines(keepends=True))
    text = text[:start] + replacement + text[end:]
    anchor = "class DataLoader3D(SlimDataLoaderBase):"
    text = _replace_once(text, anchor, CP_MODE_SOURCE + "\n\n" + anchor, "DataLoader3D declaration")
    text = _replace_once(text, '            self.cp_configs = cp_configs\n',
                         '''            self.cp_configs = cp_configs
        if self.cp_configs.get('paper_profile', False):
            # Immutable donor availability is verified once, not 168 NFS stats
            # on every online CP event. The row set remains all 168 patients.
            self._paper_tumor_files = {}
            self._paper_empty_tumor_ids = set()
            missing = []
            for key in self.list_of_keys:
                filename = self._data[key]['data_file'][:-4] + "_tumor.npy"
                if os.path.isfile(filename):
                    self._paper_tumor_files[key] = filename
                elif self.cp_configs.get('allow_empty_intra', False):
                    record = self._data[key]
                    props = record.get('properties')
                    if props is None:
                        props = load_pickle(record['properties_file'])
                    locations = props.get('class_locations', {}).get(TUMOR_LABEL)
                    if locations is not None and len(locations) == 0:
                        self._paper_empty_tumor_ids.add(key)
                    else:
                        missing.append(key)
                else:
                    missing.append(key)
            if missing:
                raise RuntimeError("Training patients lack extracted tumors: " + str(missing))
            if not self._paper_tumor_files:
                raise RuntimeError("No actual training tumor is available for TumorCP")
''', 'static paper donor inventory')
    # Limit these additions to the active CP-enabled DataLoader3D class.
    loader_start = text.index('class DataLoader3D(SlimDataLoaderBase):')
    head, body = text[:loader_start], text[loader_start:]
    body = _replace_once(body, '        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)\n',
                         '        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)\n        paper_cp_audit = []\n',
                         'CP batch audit initialization')
    body = _replace_once(body, "        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}",
                         "        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys, 'cp_audit': paper_cp_audit}",
                         'CP batch audit metadata')
    body = _replace_once(body, "        if self.cp_configs['do_inter_cp'] and self.cp_configs['do_match']:\n",
                         "        self._paper_last_paste_voxels = int(np.count_nonzero(seg_patch == TUMOR_LABEL))\n        if self.cp_configs['do_inter_cp'] and self.cp_configs['do_match']:\n",
                         'actual clipped paste mask audit')
    text = head + body
    old = """                                                               do_rotation=cp_configs['do_rotation'], p_rot=cp_configs['p_rot'],
                                                               do_elastic=cp_configs['do_elastic'], p_elas=cp_configs['p_eldef'])"""
    new = """                                                               do_rotation=cp_configs['do_rotation'], p_rot=cp_configs['p_rot'],
                                                               do_elastic=cp_configs['do_elastic'], p_elas=cp_configs['p_eldef'],
                                                               scale=cp_configs.get('paper_scale_range', (0.75, 1.25)),
                                                               angle=cp_configs.get('paper_angle_range', (0, 2 * np.pi)),
                                                               alpha=cp_configs.get('paper_elastic_alpha', (0., 1000.)),
                                                               sigma=cp_configs.get('paper_elastic_sigma', (10., 13.)))"""
    return _replace_once(text, old, new, "active spatial augmentation arguments")


def build_runtime(upstream: Path, destination: Path) -> dict:
    """Copy ``upstream/nnunet`` into a new destination and return its receipt.

    Existing destinations are verified byte-for-byte against their receipt;
    unrelated or modified files are rejected rather than replaced.
    """
    upstream, destination = Path(upstream).resolve(), Path(destination).resolve()
    source = upstream / "nnunet"
    receipt_path = destination / "runtime_receipt.json"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination == upstream or upstream in destination.parents:
        raise ValueError("Runtime destination must be outside the preserved upstream checkout")
    if destination.exists():
        if not receipt_path.is_file():
            raise FileExistsError(f"Runtime exists without a completion receipt: {destination}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for relative, digest in receipt['runtime_sha256'].items():
            path = destination / relative
            if not path.is_file() or _sha(path) != digest:
                raise ValueError(f"Existing runtime changed: {path}")
        current = {p.relative_to(source).as_posix(): _sha(p) for p in source.rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts}
        if current != receipt['upstream_sha256']:
            raise ValueError("Preserved upstream bytes differ from the completed runtime receipt")
        if _sha(Path(__file__).with_name('paper_trainer.py')) != receipt['paper_trainer_sha256']:
            raise ValueError("Paper trainer changed since this runtime was built")
        if _sha(Path(__file__)) != receipt['builder_sha256']:
            raise ValueError("Runtime builder changed; use a new isolated runtime destination")
        return receipt
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, destination / 'nnunet', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    original = {p.relative_to(source).as_posix(): _sha(p) for p in source.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
    patches = []
    for path in sorted((destination / 'nnunet').rglob('*.py')):
        before = path.read_text(encoding='utf-8')
        after = before
        reasons = []
        if path.name == 'paths.py' and path.parent == destination / 'nnunet':
            after = _patch_paths(after)
            reasons.append('isolated environment paths')
        if path.relative_to(destination).as_posix() == 'nnunet/training/dataloading/dataset_loading.py':
            after = _patch_loader(after)
            reasons.extend(['single CP event and distinct training-only inter donor',
                            'explicit paper spatial parameters'])
        if path.name == 'extract_tumors.py':
            after = _replace_once(after, 'class TumorExtractor(object):',
                                  PACK_TUMORS_SOURCE + '\n\nclass TumorExtractor(object):',
                                  'extracted tumor container helper')
            after = _replace_once(after, 'np.array(tumors)', '_paper_pack_tumors(tumors)',
                                  'ragged extracted tumor container')
            reasons.append('ragged tumor container with original numeric ndarray voxel dtypes')
        after, aliases = re.subn(r'\bnp\.(int|float|bool|complex)\b', lambda m: m.group(1), after)
        if aliases:
            reasons.append(f'modern NumPy removed scalar aliases: {aliases}')
        if path.relative_to(destination).as_posix() in (
                'nnunet/training/network_training/network_trainer.py',
                'nnunet/training/model_restore.py'):
            # These files are our own training artifacts. PyTorch >=2.6 changed
            # the default and otherwise rejects their NumPy optimizer metadata.
            after, torch_loads = re.subn(
                r"torch\.load\((fname|i), map_location=torch\.device\('cpu'\)\)",
                r"torch.load(\1, map_location=torch.device('cpu'), weights_only=False)", after)
            if torch_loads:
                reasons.append(f'explicit trusted-own-checkpoint PyTorch loading: {torch_loads}')
        if after != before:
            path.write_text(after, encoding='utf-8', newline='\n')
            patches.append({'path': path.relative_to(destination).as_posix(), 'reasons': reasons,
                            'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                            'after_sha256': _sha(path)})
        ast.parse(after, filename=str(path))
    trainer = Path(__file__).with_name('paper_trainer.py')
    target = destination / 'nnunet/training/network_training/nnUNetTrainerV2_TumorCPPaper.py'
    shutil.copyfile(trainer, target)
    ast.parse(target.read_text(encoding='utf-8'))
    receipt = {
        'format': 'tumorcp_kits19_isolated_paper_runtime_v1',
        'profile': 'ImgDA + TumorCPstar, paper event/transform settings',
        'upstream': str(upstream), 'runtime': str(destination),
        'upstream_sha256': original, 'paper_trainer_sha256': _sha(trainer),
        'builder_sha256': _sha(Path(__file__)),
        'patches': patches, 'patch_file_count': len(patches),
        'runtime_sha256': {p.relative_to(destination).as_posix(): _sha(p)
                           for p in sorted((destination / 'nnunet').rglob('*')) if p.is_file()},
        'paper_exact_split_ids_published': False,
        'split': 'Explicit TUMORCP_PAPER_DATASET_JSON profile when supplied; otherwise official KiTS19 KFold5 seed12345 fold0',
        'training_started': False, 'full_evaluation_complete': False,
        'quality_verified': False,
    }
    with receipt_path.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)
    return receipt
