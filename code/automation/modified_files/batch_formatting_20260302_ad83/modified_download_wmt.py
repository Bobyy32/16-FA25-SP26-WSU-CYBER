#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language corpus fetching utilities.
Fetch and preprocess multilingual datasets for downstream NLP workflows.

Inline comments relocated to function documentation:
- Source/target file naming conventions follow validation/source/val split patterns
- Progress tracking prevents I/O blocking on slower storage media
- Path objects enable safer directory operations with automatic cleanup
"""

from pathlib import Path

import fire
from tqdm import tqdm


def fetch_language_corpus_archive(src_lang="ro", tgt_lang="en", dataset="wmt16", save_dir=None) -> None:
    """Retrieve language pair corpus from specified WMT dataset.
    
    Fetch multilingual datasets and organize into source/target splits following validation, training, and testing conventions.
    Output files stored in directory structure: val.source, val.target, train.source, train.target, test.source, test.target
    
    Args:
        src_lang: Source language identifier (e.g., 'ro' for Romanian)
        tgt_lang: Target language identifier (e.g., 'en' for English)
        dataset: Dataset source identifier (wmt16, wmt17, etc.)
        save_dir: Output directory path or None for auto-generated naming
    
    Usage:
        >>> fetch_language_corpus_archive('ro', 'en', dataset='wmt16') # saves to wmt16-ro-en
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    language_tuple = f"{src_lang}-{tgt_lang}"
    print(f"Converting {dataset}-{language_tuple}")
    language_corpus_obj = datasets.load_dataset(dataset, language_tuple)
    if save_dir is None:
        save_dir = f"{dataset}-{language_tuple}"
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for split_name in language_corpus_obj:
        print(f"Splitting {split_name} with {language_corpus_obj[split_name].num_rows} records")

        # Split naming convention follows standard dataset conventions (validation -> val)
        split_identifier = "val" if split_name == "validation" else split_name
        source_file_reference = save_dir.joinpath(f"{split_identifier}.source")
        target_file_reference = save_dir.joinpath(f"{split_identifier}.target")
        source_file_pointer = source_file_reference.open("w+")
        target_file_pointer = target_file_reference.open("w+")

        # Progress bar tracking avoids blocking on slower storage devices
        for current_record in tqdm(language_corpus_obj[split_name]):
            translation_example = current_record["translation"]
            source_file_pointer.write(translation_example[src_lang] + "\n")
            target_file_pointer.write(translation_example[tgt_lang] + "\n")

    print(f"Saved {dataset} dataset to {save_dir}")


if __name__ == "__main__":
    fire.Fire(fetch_language_corpus_archive)