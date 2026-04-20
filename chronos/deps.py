"""
chronos/deps.py — Automatic dependency bootstrap for minimind-master kernel.

When minimind-master is not found locally, this module downloads it from
GitHub and makes it importable. Called automatically on first import of
any chronos.model module.

Legal notice: minimind is licensed under Apache-2.0.
See THIRD_PARTY_NOTICES.md for full attribution.
"""
import sys
import os
import subprocess
import importlib
from pathlib import Path

MINIMIND_REPO = "https://github.com/jingyaogong/minimind"
MINIMIND_COMMIT = "master"  # pin to a tag/commit for reproducibility

# Candidate local paths (in priority order)
_CANDIDATE_PATHS = [
    Path(__file__).parent.parent.parent / "minimind-master",   # sibling dir
    Path(__file__).parent.parent / "minimind-master",
    Path.home() / ".cache" / "chronos" / "minimind-master",
]


def _find_local() -> Path | None:
    for p in _CANDIDATE_PATHS:
        if (p / "model" / "model_minimind.py").exists():
            return p
    return None


def _download(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Chronos] minimind not found locally. Downloading from {MINIMIND_REPO} ...")
    print(f"[Chronos] Target: {target}")
    print(f"[Chronos] License: Apache-2.0  |  Copyright jingyaogong")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", MINIMIND_COMMIT,
         MINIMIND_REPO, str(target)],
        check=True,
    )
    print(f"[Chronos] minimind downloaded to {target}")
    return target


def ensure_minimind() -> Path:
    """
    Ensure minimind-master is available and on sys.path.
    Returns the path to the minimind root directory.
    """
    local = _find_local()
    if local is None:
        cache_dir = Path.home() / ".cache" / "chronos" / "minimind-master"
        local = _download(cache_dir)

    root = str(local)
    if root not in sys.path:
        sys.path.insert(0, root)
    return local


def get_tokenizer_path() -> str:
    """Return the path to the minimind tokenizer directory."""
    root = ensure_minimind()
    return str(root / "model")


# Run on import
_minimind_root = ensure_minimind()
