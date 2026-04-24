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


def download_wmt_dataset(source_language="ro", target_language="en", data_source="wmt16", storage_directory=None):
    """Retrieve a dataset using datasets package and store it in expected format for finetune.py workflow

    Output directory structure: train.source, train.target, val.source, val.target, test.source, test.target.

    Parameters:
        source_language: <str> origin language identifier
        target_language: <str> destination language identifier  
        data_source: <str> wmt16, wmt17, etc. starting with wmt16 is recommended for smaller scale. Get complete list via `import datasets; print([d.id for d in datasets.list_datasets() if "wmt" in d.id])`
        storage_directory: <str>, destination path for datasets, defaults to f'{data_source}-{source_language}-{target_language}'

    Usage:
        >>> download_wmt_dataset('ro', 'en', data_source='wmt16') # saves to wmt16-ro-en
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    pair = f"{source_language}-{target_language}"
    print(f"Converting {data_source}-{pair}")
    ds = datasets.load_dataset(data_source, pair)
    if storage_directory is None:
        storage_directory = f"{data_source}-{pair}"
    storage_directory = Path(storage_directory)
    storage_directory.mkdir(exist_ok=True)

    for split in ds:
        print(f"Splitting {split} with {ds[split].num_rows} records")

        # to save to val.source, val.target like summary datasets
        fn = "val" if split == "validation" else split
        src_path = storage_directory.joinpath(f"{fn}.source")
        tgt_path = storage_directory.joinpath(f"{fn}.target")
        src_fp = src_path.open("w+")
        tgt_fp = tgt_path.open("w+")

        # reader is the bottleneck so writing one record at a time doesn't slow things down
        for x in tqdm(ds[split]):
            ex = x["translation"]
            src_fp.write(ex[source_language] + "\n")
            tgt_fp.write(ex[target_language] + "\n")

    print(f"Saved {data_source} dataset to {storage_directory}")


if __name__ == "__main__":
    fire.Fire(download_wmt_dataset)