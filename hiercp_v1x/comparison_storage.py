"""Small present-time write checks; never a future output-capacity guarantee."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import sys
import uuid


def storage_error(error, *, operation, path):
    """Keep OS evidence, including quota versus filesystem exhaustion, visible."""
    if error.errno == getattr(errno, 'EDQUOT', 122):
        reason = 'EDQUOT: disk quota exceeded (filesystem free space does not establish remaining quota)'
    elif error.errno == errno.ENOSPC:
        reason = 'ENOSPC: filesystem space or inode allocation exhausted'
    else:
        reason = errno.errorcode.get(error.errno, type(error).__name__)
    return OSError(error.errno,
        f'{operation} failed; {reason}; {error.strerror or error}. '
        'Existing experiment outputs were not removed; restore writable storage before retrying',
        str(error.filename or path))


def report_secondary_failure(primary, secondary, *, operation, stream=None):
    """Report cleanup errors without replacing the primary exception/cause (3.10)."""
    failures = getattr(primary, 'comparison_secondary_errors', [])
    failures.append((operation, secondary))
    primary.comparison_secondary_errors = failures
    try:
        print(f'Secondary {operation} failure; primary exception preserved: '
              f'{type(secondary).__name__}: {secondary}',
              file=sys.stderr if stream is None else stream, flush=True)
    except OSError as reporting_error:
        # A broken log stream must not replace the original failure either.
        failures.append(('report secondary failure', reporting_error))


def probe_output_storage(root):
    """Create/write/flush/fsync/rename/remove only a new private 4-KiB probe.

    This detects an inability to write now, not capacity for future checkpoints
    or cache growth. The private directory prevents rename from replacing an
    existing file; cleanup never recursively removes anything.
    """
    root = Path(root)
    directory = root / ('.storage_probe_' + uuid.uuid4().hex)
    path = directory / 'write.tmp'
    renamed = directory / 'renamed.tmp'
    owned_directory = False
    owned_file = None
    operation = 'create experiment output directory'
    primary = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        operation = 'create private storage probe directory'
        directory.mkdir(exist_ok=False)
        owned_directory = True
        operation = 'exclusively create storage probe'
        with path.open('xb') as stream:
            owned_file = path
            operation = 'write storage probe'
            stream.write(b'comparison output storage probe\n'.ljust(4096, b'.'))
            operation = 'flush storage probe'
            stream.flush()
            operation = 'fsync storage probe'
            os.fsync(stream.fileno())
        operation = 'rename storage probe'
        path.rename(renamed)
        owned_file = renamed
        operation = 'remove own storage probe'
        renamed.unlink()
        owned_file = None
        operation = 'remove own storage probe directory'
        directory.rmdir()
        owned_directory = False
    except OSError as error:
        target = (root if operation == 'create experiment output directory'
                  else directory if 'directory' in operation else owned_file or path)
        primary = storage_error(error, operation=operation, path=target)
        raise primary from error
    finally:
        if owned_directory:
            try:
                if owned_file is not None:
                    owned_file.unlink()
                directory.rmdir()
            except OSError as cleanup_error:
                if primary is None:
                    raise
                report_secondary_failure(primary, cleanup_error, operation='own storage probe cleanup')
    return dict(format='comparison_output_write_probe_v1', output=str(root), bytes_written=4096,
                operations=['exclusive_create', 'write', 'flush', 'fsync', 'rename', 'remove_own_probe'],
                scope='present-time small write only; not future checkpoint/cache capacity or quota guarantee')
