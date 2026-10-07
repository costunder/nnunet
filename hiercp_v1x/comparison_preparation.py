"""Exact execution caches for the existing four comparison arms.

The archived source, candidate inventory, sampled-view epochs, training batch,
model and optimizer contracts remain owned by their original implementations.
"""
from __future__ import annotations


def prepared_provider(original, log_path=None):
    from .comparison_inputs import local_mask_provider
    from .comparison_source_cache import source_cache_provider
    from .comparison_upper_cache import compact_upper_provider
    from .comparison_sample_cache import sample_cache_provider
    from .comparison_views import parallel_view_provider
    from .comparison_data_timing import timed_provider

    provider = sample_cache_provider(parallel_view_provider(compact_upper_provider(
        source_cache_provider(local_mask_provider(original)))))
    return provider if log_path is None else timed_provider(provider, log_path)
