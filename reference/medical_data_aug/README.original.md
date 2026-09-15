# 3D Liver Tumor Copy-Paste Augmentation

This project augments 3D liver CT volumes by copying an existing tumor from each case and pasting it into another valid location inside the same liver. The corresponding tumor label is pasted into the segmentation volume at the same location.

## Files

- `3d_copy_paste_tumor.py` — batch augmentation script.
- `view_nifti_liver.ipynb` — notebook for inspecting NIfTI files, visualizing liver/tumor labels, and experimenting with augmentation on a single case.

## Requirements

- Python 3.9 or newer
- NumPy
- NiBabel
- SciPy
- scikit-image
- Matplotlib
- Plotly
- JupyterLab or Jupyter Notebook

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy nibabel scipy scikit-image matplotlib plotly jupyterlab
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Expected Dataset Structure

Download the Liver CT dataset from the following Google Drive folder:

[External medical-data sharing link omitted from the public source archive.]

After downloading and extracting the dataset, arrange the CT images and labels as shown below. Rename or reorganize the files if necessary to match the naming convention expected by the script.

Run the code from the project directory. The input data must use this structure:

```text
project/
├── 3d_copy_paste_tumor.py
├── view_nifti_liver.ipynb
└── Data/
    ├── image/
    │   ├── liver_1_0000.nii.gz
    │   ├── liver_2_0000.nii.gz
    │   └── ...
    └── labels/
        ├── liver_1.nii.gz
        ├── liver_2.nii.gz
        └── ...
```

Each image must have a matching label:

```text
Data/image/liver_<case_id>_0000.nii.gz
Data/labels/liver_<case_id>.nii.gz
```

The default label definitions are:

- `0`: background
- `1`: liver
- `2`: tumor

Change `LIVER_LABEL` and `TUMOR_LABEL` in the script if the dataset uses different values.

## Run Batch Augmentation

From the project directory, run:

```bash
python 3d_copy_paste_tumor.py
```

The script processes every file matching:

```text
Data/image/liver_*_0000.nii.gz
```

It creates the following output structure automatically:

```text
Data_aug/
├── image/
│   ├── liver_1_0000.nii.gz
│   └── ...
└── labels/
    ├── liver_1.nii.gz
    └── ...
```

The augmented CT and label retain their original affine matrices and NIfTI headers. If both output files for a case already exist, that case is skipped to avoid overwriting the previous result.

## Main Configuration

Edit the configuration section near the top of `3d_copy_paste_tumor.py` before running it:

| Setting | Default | Description |
|---|---:|---|
| `DATA_DIR` | `"Data"` | Input dataset directory |
| `DATA_AUG_DIR` | `"Data_aug"` | Output dataset directory |
| `LIVER_LABEL` | `1` | Liver label value |
| `TUMOR_LABEL` | `2` | Tumor label value |
| `NUM_COPIES` | `1` | Number of new tumors requested per case |
| `BLEND_BORDER` | `0` | Edge feathering width in voxels; `0` uses hard copy-paste |
| `INTENSITY_SCALE_RANGE` | `(0.95, 1.05)` | Random multiplicative intensity change |
| `INTENSITY_SHIFT_RANGE` | `(-5.0, 5.0)` | Random additive intensity change |
| `MIN_LIVER_COVERAGE` | `0.85` | Minimum fraction of the pasted tumor inside the liver |
| `OCCUPIED_CLEARANCE_VOX` | `2` | Minimum voxel margin around existing and pasted tumors |
| `MIN_CENTER_SEPARATION_VOX` | `12` | Minimum center distance from occupied tumor regions |
| `RNG_SEED` | `None` | Random seed; use an integer for reproducible runs |

Example reproducible configuration:

```python
NUM_COPIES = 2
BLEND_BORDER = 2
RNG_SEED = 42
```

If the script cannot find a valid location, it prints a warning. In that situation, reduce `MIN_LIVER_COVERAGE`, `OCCUPIED_CLEARANCE_VOX`, or `MIN_CENTER_SEPARATION_VOX`. The case may still be saved even when fewer tumors than requested were placed, so check the reported `pasted` count.

## Use the Visualization Notebook

Start JupyterLab:

```bash
jupyter lab
```

Open `view_nifti_liver.ipynb`, then update the paths in its configuration cells, for example:

```python
IMAGE_PATH = "Data/image/liver_1_0000.nii.gz"
LABEL_PATH = "Data/labels/liver_1.nii.gz"
```

The notebook includes:

- image shape, spacing, data type, and unique-label inspection;
- axial, coronal, and sagittal views with label overlays;
- 3D liver and tumor surface visualization;
- interactive Plotly visualization;
- single-case copy-paste experimentation; and
- saving augmented single-case NIfTI volumes.

Run only the section needed, because later cells contain independent examples with their own file paths and configuration. In particular, replace hard-coded case paths such as `liver_70` before running those cells. Some 3D volume plots use substantial memory; keep `show_ct_volume = False` or increase `downsample` for large volumes.

## How the Augmentation Works

For each case, the batch script:

1. Loads the CT image and segmentation as 3D arrays.
2. Finds connected tumor components and randomly selects one source tumor.
3. Crops the source CT and tumor-mask patch.
4. Searches for a new location inside the liver.
5. Rejects locations that violate liver coverage, overlap, clearance, or center-separation constraints.
6. Applies random intensity scaling and shifting to the copied CT patch.
7. Pastes the tumor into both the CT and segmentation volumes.
8. Saves the augmented image-label pair.

For 4D inputs, the loader selects the first channel and processes it as a 3D volume.

## Console Messages

- `[OK]` — the case was processed and saved.
- `[Warn]` — one requested tumor copy could not be placed.
- `[Skip] No liver` or `[Skip] No tumor` — the required label is absent.
- `[Skip] Missing label` — no matching label file was found.
- `[Skip] Already exists` — the output pair already exists.
- `[Error]` — the case failed; the remaining cases continue processing.

The final line reports the number of processed, successful, and skipped cases.

## Re-running Cases

To regenerate a case, remove or rename both of its existing output files before running the script again:

```text
Data_aug/image/liver_<case_id>_0000.nii.gz
Data_aug/labels/liver_<case_id>.nii.gz
```

Keep the original `Data/` directory unchanged and review augmented image-label alignment before using the generated data for model training.
