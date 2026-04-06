```python
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


def fetch_training_corpus(src_lang="ro", tgt_lang="en", dataset_name="wmt16", output_directory=None) -> None:
    """Retrieve a training collection via datasets package and persist in structure expected by finetune.py
    Save directory format: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        src_lang: <str> input language source
        tgt_lang: <str> target language destination
        dataset_name: <str> wmt16, wmt17, etc. wmt16 is a good start as it's small. To get the full list run `import datasets; print([d.id for d in datasets.list_datasets() if "wmt" in d.id])`
        output_directory: <str>, where to save the datasets, defaults to f'{dataset_name}-{src_lang}-{tgt_lang}'

    Usage:
        >>> fetch_training_corpus('ro', 'en', dataset_name='wmt16') # saves to wmt16-ro-en
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    pair = f"{src_lang}-{tgt_lang}"
    print(f"Converting {dataset_name}-{pair}")
    data_source = datasets.load_dataset(dataset_name, pair)
    if output_directory is None:
        output_directory = f"{dataset_name}-{pair}"
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    for partition in data_source:
        print(f"Processing {partition} with {data_source[partition].num_rows} records")

        # to save to val.source, val.target like summary datasets
        segment = "validation" if partition == "validation" else partition
        input_file = output_directory.joinpath(f"{segment}.source")
        output_file = output_directory.joinpath(f"{segment}.target")
        src_fp = input_file.open("w+")
        tgt_fp = output_file.open("w+")

        # reader is the bottleneck so writing one record at a time doesn't slow things down
        for x in tqdm(data_source[partition]):
            content = x["translation"]
            src_fp.write(content[src_lang] + "\n")
            tgt_fp.write(content[tgt_lang] + "\n")

    print(f"Saved {dataset_name} dataset to {output_directory}")


if __name__ == "__main__":
    fire.Fire(fetch_training_corpus)