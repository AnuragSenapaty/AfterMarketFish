"""Path utilities shared across the AfterMarketFish codebase."""

from __future__ import annotations

from pathlib import Path


def resolve_repo_path(path: str | Path) -> str:
    """Resolve *path* relative to the repository root if it is not absolute.

    We assume the repo root is two levels above this file
    (AfterMarketFish/aftermarketfish/).
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)

    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / p).resolve())
