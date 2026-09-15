"""Export the current text sources, not a falsely labelled historical commit.

The output is a documentation artifact only. Medical data, environments, experiment
outputs, attachments and old code bundles are never included. Untracked source
files must be selected explicitly by the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone


EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", "work", "results",
                  "feedback", "Data", "Data_aug", "node_modules"}
EXCLUDED_NAMES = {"code.txt", ".env"}


def verify_source_only_notebook(content: str, relative: str) -> None:
    """Refuse rendered patient data, rather than silently stripping on export.

    Only source cells and empty metadata are accepted. Review/sanitize the
    notebook separately; the exporter never modifies its input notebook.
    """
    notebook = json.loads(content)
    if (not isinstance(notebook, dict) or notebook.get("nbformat") != 4
            or notebook.get("metadata") != {}
            or not isinstance(notebook.get("cells"), list)
            or set(notebook) != {"cells", "metadata", "nbformat", "nbformat_minor"}):
        raise ValueError(f"Not a source-only notebook: {relative}")
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            raise ValueError(f"Malformed source notebook cell: {relative}")
        kind = cell.get("cell_type")
        allowed = {"cell_type", "metadata", "source", "id"}
        if kind == "code":
            allowed.update({"outputs", "execution_count"})
            if cell.get("outputs") != [] or cell.get("execution_count") is not None:
                raise ValueError(f"Notebook output/execution data must be cleared: {relative}")
        elif kind != "markdown":
            raise ValueError(f"Unsupported source notebook cell type: {relative}")
        source = cell.get("source")
        if (cell.get("metadata") != {} or not set(cell).issubset(allowed)
                or not (isinstance(source, str) or
                        isinstance(source, list) and all(isinstance(line, str) for line in source))):
            raise ValueError(f"Notebook attachments/metadata or malformed source: {relative}")


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root)


def source_record(root: Path, relative: str) -> tuple[dict, str]:
    name = Path(relative)
    if (name.is_absolute() or ".." in name.parts
            or any(part in EXCLUDED_PARTS for part in name.parts)
            or name.name in EXCLUDED_NAMES or name.name.startswith(".env.")):
        raise ValueError(f"Excluded/non-local handoff source: {relative}")
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Handoff source must be a regular in-project file: {relative}")
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ValueError(f"Binary input is not a handoff text source: {relative}")
    content = raw.decode("utf-8").replace("\r\n", "\n")
    if name.suffix.lower() == ".ipynb":
        verify_source_only_notebook(content, relative)
    exported = content.encode("utf-8")
    return ({"path": name.as_posix(), "source_sha256": hashlib.sha256(raw).hexdigest(),
             "export_sha256": hashlib.sha256(exported).hexdigest(),
             "export_bytes": len(exported)}, content)


def export(root: Path, output: Path, additions: list[str]) -> dict:
    root, output = root.resolve(), output.absolute()
    if output.parent.resolve() != root or output.name != "code.txt" or output.is_symlink():
        raise ValueError("Only the repository-root code.txt documentation may be replaced")
    tracked = [value.decode("utf-8") for value in git(root, "ls-files", "-z").split(b"\0") if value]
    selected = []
    for relative in sorted(set(tracked + additions)):
        name = Path(relative)
        if (any(part in EXCLUDED_PARTS for part in name.parts)
                or name.name in EXCLUDED_NAMES or name.name.startswith(".env.")):
            continue
        # Tracked removals belong to the working tree, not this snapshot.
        if relative in tracked and not (root / name).exists():
            continue
        selected.append(source_record(root, relative))
    if not selected or "gpt_handoff.md" not in {row[0]["path"] for row in selected}:
        raise ValueError("The complete handoff must include gpt_handoff.md")
    manifest = {
        "format": "hiercp_working_tree_code_export_v1",
        "base_commit": git(root, "rev-parse", "HEAD").decode("ascii").strip(),
        "snapshot": "current working tree; not a claim that changes were committed or pushed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(selected), "files": [row for row, _ in selected],
    }
    chunks = ["# HierCP complete current-source handoff\n\n",
              "Read gpt_handoff.md for active architecture, evidence and limits.\n",
              "Debug tests are not full medical/GPU training or evaluation.\n",
              "Medical data, experiment results, environments and audit attachments are excluded.\n",
              "Text is UTF-8 with LF line endings. export_bytes excludes the separator newline.\n\n",
              "## Source manifest\n\n", json.dumps(manifest, ensure_ascii=False, indent=2), "\n\n"]
    for index, (row, content) in enumerate(selected):
        ending = "\n" if index + 1 == len(selected) else "\n\n"
        chunks.extend([f"===== FILE: {row['path']} =====\n", content,
                       f"\n===== END FILE: {row['path']} ====={ending}"])
    payload = "".join(chunks).encode("utf-8")
    # Verify the exact framed content before publishing the documentation.
    offset = payload.index(b"===== FILE: ")
    for index, (row, _) in enumerate(selected):
        marker = f"===== FILE: {row['path']} =====\n".encode("utf-8")
        if payload[offset:offset + len(marker)] != marker:
            raise ValueError("Export framing verification failed")
        offset += len(marker)
        body = payload[offset:offset + row["export_bytes"]]
        if hashlib.sha256(body).hexdigest() != row["export_sha256"]:
            raise ValueError("Export source digest verification failed")
        offset += len(body)
        ending = "\n" if index + 1 == len(selected) else "\n\n"
        end = f"\n===== END FILE: {row['path']} ====={ending}".encode("utf-8")
        if payload[offset:offset + len(end)] != end:
            raise ValueError("Export end framing verification failed")
        offset += len(end)
    if offset != len(payload):
        raise ValueError("Unexpected trailing export content")
    with tempfile.NamedTemporaryFile(mode="wb", prefix=".code-export-", suffix=".tmp",
                                     dir=root, delete=False) as stream:
        staging = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if staging.read_bytes() != payload:
        raise OSError(f"Staging verification failed; preserved: {staging}")
    os.replace(staging, output)
    return {"path": str(output), "file_count": len(selected), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "base_commit": manifest["base_commit"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-untracked", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(export(args.project_root, args.project_root / "code.txt",
                            args.include_untracked), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
