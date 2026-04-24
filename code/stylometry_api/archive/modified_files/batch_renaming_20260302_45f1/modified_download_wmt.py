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


def process_wmt_data(source_lang="ro", target_lang="en", dataset_name="wmt16", output_path=None) -> None:
    """Fetch and format dataset for training using datasets package
    Output format: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        source_lang: <str> input language
        target_lang: <str> output language
        dataset_name: <str> wmt16, wmt17, etc. wmt16 is a good start as it's small. To get the full list run `import datasets; print([d.id for d in datasets.list_datasets() if "wmt" in d.id])`
        output_path: <str>, where to store the datasets, defaults to f'{dataset_name}-{source_lang}-{target_lang}'

    Usage:
        >>> process_wmt_data('ro', 'en', dataset_name='wmt16') # saves to wmt16-ro-en
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")
    
    lang_pair = f"{source_lang}-{target_lang}"
    print(f"Processing {dataset_name}-{lang_pair}")
    
    data_set = datasets.load_dataset(dataset_name, lang_pair)
    
    if output_path is None:
        output_path = f"{dataset_name}-{lang_pair}"
        
    save_location = Path(output_path)
    save_location.mkdir(exist_ok=True)

    # Process each dataset split
    for split_name in data_set:
        print(f"Processing {split_name} with {data_set[split_name].num_rows} entries")

        # Map validation to val for consistency
        file_prefix = "val" if split_name == "validation" else split_name
        source_file = save_location.joinpath(f"{file_prefix}.source")
        target_file = save_location.joinpath(f"{file_prefix}.target")
        
        src_handle = source_file.open("w+")
        tgt_handle = target_file.open("w+")

        # Efficient processing using list comprehension approach
        [src_handle.write(x["translation"][source_lang] + "\n") or 
         tgt_handle.write(x["translation"][target_lang] + "\n") 
         for x in tqdm(data_set[split_name])]

    print(f"Stored {dataset_name} dataset to {save_location}")


if __name__ == "__main__":
    fire.Fire(process_wmt_data)