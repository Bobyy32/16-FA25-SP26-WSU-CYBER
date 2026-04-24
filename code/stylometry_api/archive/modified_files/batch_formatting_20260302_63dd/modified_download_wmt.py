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


def download_wmt_dataset(src_lang: str = 'ro', tgt_lang: str = 'en', dataset: str = 'wmt16', save_dir=None) -> None:
    """Download WMT corpus using datasets package and store in fine-tune format.

    Args:
        src_lang: <str> source language identifier
        tgt_lang: <str> target language identifier  
        dataset: <str> wmt16, wmt17 etc.; try wmt16 first as it's compact. List options with `import datasets; print([d.id for d in datasets.list_datasets() if "wmt" in d.id])`
        save_dir: <str|None> directory location for datasets, defaults to f'{dataset}-{src_lang}-{tgt_lang}'

    Notes:
        Validation files use 'val' prefix matching summary dataset conventions.
        Reader performance bottleneck means single-record writes don't impact speed.

    Usage:
        >>> download_wmt_dataset('ro', 'en', dataset='wmt16') # saves to wmt16-ro-en
    """
    pair = f"{src_lang}-{tgt_lang}"
    print(f"Converting {dataset}-{pair}")
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    ds = datasets.load_dataset(dataset, pair)
    if save_dir is None or not isinstance(save_dir, Path):
        save_dir = Path(str(save_dir) if save_dir else f"{dataset}-{pair}")
    save_dir.mkdir(exist_ok=True)

    for split in tqdm(ds.keys(), desc="splits"):
        print(f"Splitting {split} with {ds[split].num_rows} records")
        fn = "val" if split == "validation" else split
        src_file = save_dir.joinpath(f"{fn}.source")
        tgt_file = save_dir.joinpath(f"{fn}.target")

        # reader is the bottleneck so writing one record at a time doesn't slow things down
        with open(src_file, "w+") as src_fp, open(tgt_file, "w+") as tgt_fp:
            for x in tqdm(ds[split], desc="records"):
                ex = x["translation"]
                src_fp.write(ex[src_lang] + "\n")
                tgt_fp.write(ex[tgt_lang] + "\n")

    print(f"Saved {dataset} dataset to {save_dir}")


if __name__ == "__main__":
    fire.Fire(download_wmt_dataset)