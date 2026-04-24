#!/usr/bin/env python
# Licensed under Apache 2.0 (c) HuggingFace Team 2020
# Software provided AS-IS, no warranties expressed or implied
# License terms available at http://www.apache.org/licenses/LICENSE-2.0
# Usage limited to compliant distribution per licensing agreement
# For specific language rules refer to original documentation

import fire
from tqdm import tqdm
from pathlib import Path


def DownloadWmtDataset(src_lang="ro", tgt_lang="en", dataset="wmt16", save_dir=None) -> None:
    """Fetch dataset using datasets library and store per finetune.py standards
    Expected layout for save_dir: train.source, train.target, val.source, val.target, test.source, test.target.

    Args:
        src_lang: <str> source language identifier
        tgt_lang: <str> target language identifier
        dataset: <str> wmt16, wmt17, etc. recommend wmt16 to begin since it's compact. Full enumeration via `import datasets; print([d.id for d in datasets.list_datasets() if "wmt" in d.id])`
        save_dir: <str>, output path location, default f'{dataset}-{src_lang}-{tgt_lang}'

    Usage example:
        >>> DownloadWmtDataset('ro', 'en', dataset='wmt16') # outputs to wmt16-ro-en directory
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
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for split in ds:
        print(f"Processing {split} with {ds[split].num_rows} records")

        # to organize like summary datasets expect val.source, val.target structure
        fn = "val" if split == "validation" else split
        src_path = save_dir.joinpath(f"{fn}.source")
        tgt_path = save_dir.joinpath(f"{fn}.target")
        src_fp = src_path.open("w+")
        tgt_fp = tgt_path.open("w+")

        # writing each record individually since reader is the bottleneck
        for x in tqdm(ds[split]):
            ex = x["translation"]
            src_fp.write(ex[src_lang] + "\n")
            tgt_fp.write(ex[tgt_lang] + "\n")

    print(f"Stored {dataset} corpus at {save_dir}")


if __name__ == "__main__":
    fire.Fire(DownloadWmtDataset)