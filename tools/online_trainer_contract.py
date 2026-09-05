"""Fingerprint required, externally supplied OnlineCP trainer implementations."""
from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any, Sequence


def trainer_source_identity(module_name: str, class_names: Sequence[str]) -> dict[str, Any]:
    """Fail if the real trainers or their readable Python sources are absent.

    The benchmark compares this record on both completion and reuse. Hashing
    base-class sources also detects changes to inherited training behaviour.
    This function neither installs nor substitutes a trainer implementation.
    """
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Trainer contract requires distinct, nonempty class names")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Required OnlineCP trainer or dependency is unavailable: {module_name}. "
            "The original code snapshot does not include the custom trainer files. "
            "Install the original server-side implementation and its dependencies "
            "before running or reusing the online experiment."
        ) from exc

    objects: list[Any] = [module]
    qualified_classes: dict[str, str] = {}
    for name in class_names:
        trainer = getattr(module, name, None)
        if not inspect.isclass(trainer):
            raise RuntimeError(f"Required trainer class is missing: {module_name}.{name}")
        qualified_classes[name] = f"{trainer.__module__}.{trainer.__qualname__}"
        objects.extend(base for base in trainer.__mro__ if base.__module__ != "builtins")

    sources: dict[str, str] = {}
    for obj in objects:
        try:
            filename = inspect.getsourcefile(obj)
        except (TypeError, OSError) as exc:
            raise RuntimeError(f"Cannot locate required trainer source: {obj!r}") from exc
        if filename is None:
            raise RuntimeError(f"Required trainer source is unavailable: {obj!r}")
        path = Path(filename).resolve(strict=True)
        key = path.as_posix()
        if key in sources:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        sources[key] = digest.hexdigest()
    return {
        "format": "hiercp_online_trainer_sources_v1",
        "module": module_name,
        "classes": qualified_classes,
        "source_sha256": dict(sorted(sources.items())),
    }
