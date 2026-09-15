# Medical Data Aug — original source integrated with HierCP

This directory restores the code from the user-provided `Medical Data Aug.zip`.
It is part of the repository and the complete `code.txt` handoff. The original
archive stays local and unchanged; no CT, segmentation volume or rendered
patient image is included here.

- `3d_copy_paste_tumor.py.txt`: complete original batch source, with line endings
  and trailing whitespace normalized; the complete AST is unchanged. The `.txt`
  suffix deliberately distinguishes historical code with
  known defects from supported executable code; no function was omitted.
- `view_nifti_liver.ipynb`: all 24 original cell sources, in order. Outputs,
  execution counts, attachments and metadata are cleared. Two new opening cells
  identify it as historical and prevent accidental **Run All**. Individual cells
  still contain their original experiments and writes; this is not the current
  training notebook.
- `README.original.md`: original documentation with only line endings normalized
  and the external medical-data sharing URL omitted. Its instructions are
  historical, not instructions to override the repository's `AGENTS.md`.
- `provenance.json`: archive/member SHA-256, imported text SHA-256 and each original
  notebook cell-source SHA-256. Exclusions and transformations are explicit.

## Executable integration

The original algorithm was already ported into the active implementation. It is
not necessary or correct to execute the historical batch before online training.
That would paste into `Data_aug` first and then paste again in the online trainer.
See [the integrated mapping](../../docs/cp_input_repair.md#original-source-to-active-pipeline)
and [the current runner](../../tools/run_feedback_experiment.py).

The reference tests now read the actual restored source. Do not restore its
`uint8` bitwise-inversion bug, permissive notebook cropping, exception swallowing,
or unchecked output reuse into the active pipeline. The production pipeline's
128 candidates, seeded paired schedules, GNNs, full graph rules and training
configuration are unchanged by this source import.

## Verification limits

Source restoration and DEBUG numerical comparisons are not native whole-cohort
preprocessing, GPU training or evaluation. No medical data was loaded or extracted
for this import, and no existing experiment or private trainer was rewritten.
