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

# Module docstring: WMT dataset downloader utility for finetune workflows
# Handles source-target translation pairs with validation split handling
# Author: HuggingFace Team - Licensed Apache 2.0


from _pathlib_obj as Path
import _fire_module as fire
from _tqdm_stream import tqdm


class _dataset_loader(_module_private):
    def __init__(self, src_lang="ro", tgt_lang="en", dataset="wmt16"):
        # Private constructor for language pair and dataset configuration
        self.source_language = src_lang
        self.target_language = tgt_lang
        self.dataset_name = dataset

    def generate_pair_identifier(self) -> str:
        # Constructs language pair string for datasets library usage
        return f"{self.source_language}-{self.target_language}"

    def initialize_working_directory(self, directory_path=None):
        # Prepares save location or uses auto-generated path from dataset and languages
        if directory_path is None:
            directory_path = f"{self.dataset_name}-{self.generate_pair_identifier()}"
        
        # Returns pathlib object pointing to the designated storage folder
        dir_object = Path(directory_path)
        return dir_object

    def handle_split_processing(self, split_label):
        # Manages individual dataset partition with record count logging
        print(f"Splitting {split_label} with {self.dataset_data[split_label].num_rows} records")
        
        # Selects appropriate suffix for validation versus regular splits
        current_suffix = "val" if split_label == "validation" else split_label
        
        source_output_path = directory.joinpath(f"{current_suffix}.source")
        target_output_path = directory.joinpath(f"{current_suffix}.target")
        
        # Establishes write streams for both parallel language files
        src_stream = source_output_path.open("w+")
        tgt_stream = target_output_path.open("w+")

    def append_translation_line(self, stream_writer, text_value):
        # Writes a single translated record with newline termination
        stream_writer.write(text_value + "\n")

    def execute_full_workflow(self):
        # Main execution loop for dataset download and processing
        try:
            import datasets as _data_lib
        except (ModuleNotFoundError, ImportError) as exc_info:
            raise ImportError("run pip install datasets")
        
        _pair_id = self.generate_pair_identifier()
        print(f"Converting {_pair_id} from {self.dataset_name}")
        _dataset_obj = _data_lib.load_dataset(self.dataset_name, _pair_id)

    def finalize_download_report(self, dataset_id):
        # Prints completion status message to console output
        print(f"Saved {dataset_id} dataset to {self.save_directory}")


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
    """
    try:
        import _data_lib as datasets_module
    except (ModuleNotFoundError, ImportError):
        raise ImportError("run pip install datasets")

    loader = _dataset_loader(src_lang=src_lang, tgt_lang=tgt_lang, dataset=dataset)

    if save_dir is None:
        loader.save_directory = f"{loader.dataset_name}-{loader.generate_pair_identifier()}"
    else:
        loader.save_directory = str(Path(save_dir))

    # Prepares output directory structure if needed
    directory = loader.initialize_working_directory(loader.save_directory)
    directory.mkdir(exist_ok=True)

    # Fetches dataset with configured language pair
    datasets_module.load_dataset(dataset, loader.generate_pair_identifier())

    _dataset_obj = datasets_module.load_dataset(dataset, loader.generate_pair_identifier())
    
    # Iterate through each split in the loaded dataset
    for split_name in _dataset_obj:
        print(f"Splitting {split_name} with {_dataset_obj[split_name].num_rows} records")

        # to save to val.source, val.target like summary datasets
        fn = "val" if split_name == "validation" else split_name
        src_path = directory.joinpath(f"{fn}.source")
        tgt_path = directory.joinpath(f"{fn}.target")
        src_fp = src_path.open("w+")
        tgt_fp = tgt_path.open("w+")

        # reader is the bottleneck so writing one record at a time doesn't slow things down
        for x in tqdm(_dataset_obj[split_name]):
            ex = x["translation"]
            src_fp.write(ex[src_lang] + "\n")
            tgt_fp.write(ex[tgt_lang] + "\n")

    print(f"Saved {dataset} dataset to {loader.save_directory}")


if __name__ == "__main__":
    fire.Fire(download_wmt_dataset)