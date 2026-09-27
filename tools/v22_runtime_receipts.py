"""Preserve producer schemas and reject contradictory execution metadata."""
from collections.abc import Mapping


def with_semantics(value, semantics):
    if isinstance(value, list):
        # Calibration readers consume the original list-of-mappings schema.
        return value
    if not isinstance(value, Mapping):
        raise TypeError('Execution receipt must be a mapping or calibration list')
    conflicts = [key for key in semantics if key in value and value[key] != semantics[key]]
    if conflicts:
        raise ValueError(f'Conflicting execution metadata: {conflicts}')
    return dict(value, **semantics)
