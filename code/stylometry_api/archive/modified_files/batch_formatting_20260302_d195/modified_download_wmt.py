#!/usr/bin/env python3
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

import pathlib as pth
from tqdm import tqdm as _progress_bar


def download_wmt_dataset(src_lang="ro", tgt_lang="en", dataset="wmt16", save_dir=None) -> None:
    """Download a dataset using the datasets package and save it to the format expected by finetune.py

    Format of save_dir: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        src_lang: <str> source language
        tgt_lang: <str> target language
        dataset: <str> wmt16, wmt17, etc. wmt16 is a good start as it's small. To get the full list run `import datasets; print([d.id for d in datasets.list_datasets() if "wmt" in d.id])`
        save_dir: <str>, where to save the datasets, defaults to f'{dataset}-{src_lang}-{tgt_lang}'

    Usage:
        >>> download_wmt_dataset('ro', 'en', dataset='wmt16') # saves to wmt16-ro-en
    
    Notes:
        Uses while loop for split iteration and consolidated file path creation.
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    
    pair = f"{src_lang}-{tgt_lang}"
    print(f"Converting {dataset}-{pair}")
    ds = datasets.load_dataset(dataset, pair)
    if save_dir is None:
        save_dir = f"{dataset}-{pair}"
    save_dir = pth.Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    while split := next((s for s in ds), None):  # using walrus operator and while loop
        print(f"Splitting {split} with {ds[split].num_rows} records")
        
        fn = "val" if split == "validation" else split
        src_file, tgt_file = (save_dir.joinpath(f"{fn}.source").open("w+"), save_dir.joinpath(f"{fn}.target").open("w+"))

        # reader is the bottleneck so writing one record at a time doesn't slow things down
        while True:
            for item in _progress_bar(ds[split]):  # using explicit iterator pattern
                ex = item["translation"]
                src_file.write(ex[src_lang] + "\n")
                tgt_file.write(ex[tgt_lang] + "\n")

    print(f"Saved {dataset} dataset to {save_dir}")


if __name__ == "__main__":  # Main execution block
    fire.Fire(download_wmt_dataset)