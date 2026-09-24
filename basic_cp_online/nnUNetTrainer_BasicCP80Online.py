"""User-authorized CP80 comparator; the original 100% trainer remains available."""
from basic_cp_online.nnUNetTrainer_OriginalBasicCPOnline import nnUNetTrainer_250epochs_OriginalBasicCPOnline


class nnUNetTrainer_250epochs_BasicCP80Online(nnUNetTrainer_250epochs_OriginalBasicCPOnline):
    cp_probability=0.8
    comparison_variant='Basic CP with shared 80% attempt gate; other original policies preserved'
