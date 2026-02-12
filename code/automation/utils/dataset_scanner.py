"""
Dataset scanner for discovering authors and files in dataset_splits/.

Handles the directory structure including the unknown/ nesting where
some authors live under dataset_splits/unknown/{author}/.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from automation.config import DATASET_DIR


def get_all_authors() -> Dict[str, Path]:
    """
    Discover all authors in the dataset_splits directory.

    Handles both direct authors (dataset_splits/{author}/) and
    nested authors (dataset_splits/unknown/{author}/).

    Returns:
        Dictionary mapping author name to their base directory path.
    """
    authors = {}

    if not DATASET_DIR.exists():
        return authors

    for entry in sorted(DATASET_DIR.iterdir()):
        if not entry.is_dir():
            continue

        if entry.name == "unknown":
            # Nested: dataset_splits/unknown/{author}/
            for sub_entry in sorted(entry.iterdir()):
                if sub_entry.is_dir() and _has_split_dirs(sub_entry):
                    authors[sub_entry.name] = sub_entry
        else:
            if _has_split_dirs(entry):
                authors[entry.name] = entry

    return authors


def _has_split_dirs(path: Path) -> bool:
    """Check if a directory has training/ or testing/ subdirectories."""
    return (path / "training").is_dir() or (path / "testing").is_dir()


def get_author_files(
    author: str,
    split: str = "testing",
) -> List[Tuple[str, Path]]:
    """
    Get all Python files for an author in a given split.

    Args:
        author: Author name.
        split: Dataset split ('testing' or 'training').

    Returns:
        List of (filename, full_path) tuples.

    Raises:
        ValueError: If author not found or split directory missing.
    """
    authors = get_all_authors()
    if author not in authors:
        raise ValueError(
            f"Author '{author}' not found. Available: {list(authors.keys())}"
        )

    split_dir = authors[author] / split
    if not split_dir.is_dir():
        raise ValueError(
            f"Split '{split}' not found for author '{author}' at {split_dir}"
        )

    files = sorted(
        (f.name, f) for f in split_dir.iterdir()
        if f.is_file() and f.suffix == ".py"
    )
    return files


def select_files_for_batch(
    author: str,
    split: str = "testing",
    count: int = 5,
    seed: Optional[int] = None,
) -> List[Tuple[str, Path]]:
    """
    Select a random subset of files for batch testing.

    Args:
        author: Author name.
        split: Dataset split ('testing' or 'training').
        count: Number of files to select.
        seed: Random seed for reproducibility.

    Returns:
        List of (filename, full_path) tuples.

    Raises:
        ValueError: If not enough files available.
    """
    all_files = get_author_files(author, split)

    if len(all_files) < count:
        raise ValueError(
            f"Author '{author}' has only {len(all_files)} files in '{split}', "
            f"but {count} were requested."
        )

    rng = random.Random(seed)
    selected = rng.sample(all_files, count)
    return sorted(selected, key=lambda x: x[0])


def resolve_author_from_path(file_path: str) -> Optional[str]:
    """
    Attempt to determine the author from a dataset file path.

    Works for paths containing dataset_splits/{author}/ or
    dataset_splits/unknown/{author}/.

    Args:
        file_path: Path to a file in dataset_splits.

    Returns:
        Author name if resolved, None otherwise.
    """
    path = Path(file_path).resolve()
    dataset_dir = DATASET_DIR.resolve()

    try:
        relative = path.relative_to(dataset_dir)
    except ValueError:
        return None

    parts = relative.parts

    if len(parts) >= 3 and parts[0] == "unknown":
        # dataset_splits/unknown/{author}/split/file.py
        return parts[1]
    elif len(parts) >= 2:
        # dataset_splits/{author}/split/file.py
        return parts[0]

    return None
