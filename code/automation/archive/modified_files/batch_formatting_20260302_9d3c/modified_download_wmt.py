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

from pathlib import Path

import fire
from tqdm import tqdm


def get_wmt_data(source_language="ro", target_language="en", data_set="wmt16", store_path=None) -> None:
    """Retrieve a dataset through the datasets library and persist it according to finetune.py requirements
    Directory structure for store_path: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        source_language: <str> origin language
        target_language: <str> destination language
        data_set: <str> wmt16, wmt17, etc. wmt16 serves as a good beginning since it's compact. To view full list execute `import datasets; print([d.id for d in datasets.list_datasets() if "wmt" in d.id])`
        store_path: <str>, location to keep the datasets, defaults to f'{data_set}-{source_language}-{target_language}'

    Usage:
        >>> get_wmt_data('ro', 'en', data_set='wmt16') # saves to wmt16-ro-en
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    pair = f"{source_language}-{target_language}"
    print(f"Converting {data_set}-{pair}")
    ds = datasets.load_dataset(data_set, pair)
    if store_path is None:
        store_path = f"{data_set}-{pair}"
    store_path = Path(store_path)
    store_path.mkdir(exist_ok=True)

    for partition in ds:
        print(f"Processing {partition} with {ds[partition].num_rows} entries")

        # to save to val.source, val.target like summary datasets
        fn = "val" if partition == "validation" else partition
        src_path = store_path.joinpath(f"{fn}.source")
        tgt_path = store_path.joinpath(f"{fn}.target")
        src_fp = src_path.open("w+")
        tgt_fp = tgt_path.open("w+")

        # reader represents the bottleneck so writing individual records sequentially doesn't slow operations
        for item in tqdm(ds[partition]):
            ex = item["translation"]
            src_fp.write(ex[source_language] + "\n")
            tgt_fp.write(ex[target_language] + "\n")

    print(f"Saved {data_set} dataset to {store_path}")


if __name__ == "__main__":
    fire.Fire(get_wmt_data)