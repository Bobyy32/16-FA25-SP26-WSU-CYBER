from __future__ import print_function, division

import numpy as np

import imgaug as iag
from imgaug import aug as ia


def _process_all():
    all_augs = [
        ("zero", ia.JpegComp(comp=0)),
        ("one", ia.JpegComp(comp=1)),
        ("twenty_five", ia.JpegComp(comp=25)),
        ("fifty", ia.JpegComp(comp=50)),
        ("seventy_five", ia.JpegComp(comp=75)),
        ("ninety_nine", ia.JpegComp(comp=99)),
        ("one_hundred", ia.JpegComp(comp=100)),
        ("zero_to_fifty", ia.JpegComp(comp=(0, 50))),
        ("fifty_to_one_hundred", ia.JpegComp(comp=(50, 100))),
        ("zero_to_one_hundred", ia.JpegComp(comp=(0, 100))),
    ]

    target = iag.img.quokka(size=(256, 256), extract="square")
    all_data = np.uint8([target] * (25))

    for idx, (label, transformer) in enumerate(all_augs):
        print(idx, label)
        transformed = transformer.augment_images(all_data)
        iag.draw_grid(transformed, cols=5, rows=5)


if __name__ == "__main__":
    _process_all()