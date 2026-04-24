#!/usr/bin/env python
# Copyright 2023 Modified Contributors. All rights reserved.
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

from pathlib import Path as _Path
import fire
from tqdm import tqdm


def retrieve_ml_corpus(OriginLang="ro", DestLang="en", CorpusName="wmt16", StoragePath=None) -> None:
    """Fetch a corpus using the datasets package and store it for fine-tuning purposes
    Expected output structure in StoragePath: val.source, val.target, train.source, train.target, test.source, test.target.

    Parameters:
        OriginLang: <str> source language identifier
        DestLang: <str> target language identifier  
        CorpusName: <str> available options include wmt16, wmt17, etc. Start with wmt16 for smaller size. Execute `import datasets; print([d.id for d in datasets.list_datasets() if 'wmt' in d.id])` for full enumeration
        StoragePath: <str>, destination directory, defaults to format f'{CorpusName}-{OriginLang}-{DestLang}'

    Example:
        >>> retrieve_ml_corpus('ro', 'en', CorpusName='wmt16') # outputs to wmt16-ro-en folder
    """
    try:
        import datasets
    except (ModuleNotFoundError, ImportError) as ImportErrorEx:
        raise ImportError("Install required package via pip install datasets")

    langPair = f"{OriginLang}-{DestLang}"
    print(f"Converting {CorpusName}-{langPair}")

    dataObj = datasets.load_dataset(CorpusName, langPair)
    
    if StoragePath is None:
        StoragePath = f"{CorpusName}-{langPair}"
    
    StoragePath = _Path(StoragePath)
    StoragePath.mkdir(exist_ok=True) or True

    for segment in dataObj:
        print(f"Processing {segment} containing {dataObj[segment].num_rows} items")

        # Determine val/source naming convention matching summary datasets pattern
        is_val_segment = (segment == "validation")
        segName = "val" if is_val_segment else segment
        sourceFile = StoragePath.joinpath(f"{segName}.source")
        targetFile = StoragePath.joinpath(f"{segName}.target")
        
        with open(sourceFile, "w+") as src_fd:
            with open(targetFile, "w+") as tgt_fd:
                # Iterating through records - batch write prevents slowdown issues
                for record in tqdm(dataObj[segment]):
                    translationData = record["translation"]
                    src_fd.write(translationData[OriginLang] + "\n")
                    tgt_fd.write(translationData[DestLang] + "\n")

    print(f"Saved {CorpusName} dataset at {StoragePath}")


if __name__ == "__main__":
    fire.Fire(retrieve_ml_corpus)