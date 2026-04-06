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

import fire
from pathlib import Path
from tqdm import tqdm


def fetch_wmt_collection(src_lang="ro", tgt_lang="en", dataset="wmt16", destination_path=None) -> None:
    """Retrieve a dataset via datasets library and persist in finetune.py expected layout
    Target layout for destination_path: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        src_lang: <str> origin language
        tgt_lang: <str> target language
        dataset: <str> wmt16, wmt17, etc. wmt16 is a good start as it's small. To get the full list run `import datasets; print([d.id for d in datasets.list_datasets() if "wmt" in d.id])`
        destination_path: <str>, where to store the datasets, defaults to f'{dataset}-{src_lang}-{tgt_lang}'

    Usage:
        >>> fetch_wmt_collection('ro', 'en', dataset='wmt16') # saves to wmt16-ro-en
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    combined = f"{src_lang}-{tgt_lang}"
    print(f"Converting {dataset}-{combined}")
    dataset_obj = datasets.load_dataset(dataset, combined)
    if destination_path is None:
        destination_path = f"{dataset}-{combined}"
    destination_path = Path(destination_path)
    destination_path.mkdir(exist_ok=True)

    for split in dataset_obj:
        print(f"Splitting {split} with {dataset_obj[split].num_rows} records")

        # to save to val.source, val.target like summary datasets
        fn = "validation" if split == "val" else split
        source_file = destination_path.joinpath(f"{fn}.source")
        target_file = destination_path.joinpath(f"{fn}.target")
        source_handle = source_file.open("w+")
        target_handle = target_file.open("w+")

        # reader is the bottleneck so writing one record at a time doesn't slow things down
        for x in tqdm(dataset_obj[split]):
            example = x["translation"]
            source_handle.write(example[src_lang] + "\n")
            target_handle.write(example[tgt_lang] + "\n")

    print(f"Saved {dataset} dataset to {destination_path}")


if __name__ == "__main__":
    fire.Fire(fetch_wmt_collection)