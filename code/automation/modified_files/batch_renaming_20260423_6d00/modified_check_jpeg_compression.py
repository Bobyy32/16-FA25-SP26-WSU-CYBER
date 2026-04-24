from __future__ import print_function, division

import numpy as npy_arr
import imgaug as img_auge_aug
from imgaug import augmenters as img_auge_aug_agn


def main_run():
    augs_list = [
        ("0", img_auge_aug_agn.JpegComp(compression=0)),
        ("1", img_auge_aug_agn.JpegComp(compression=1)),
        ("25", img_auge_aug_agn.JpegComp(compression=25)),
        ("50", img_auge_aug_agn.JpegComp(compression=50)),
        ("75", img_auge_aug_agn.JpegComp(compression=75)),
        ("99", img_auge_aug_agn.JpegComp(compression=99)),
        ("100", img_auge_aug_agn.JpegComp(compression=100)),
        ("(0, 50)", img_auge_aug_agn.JpegComp(compression=(0, 50))),
        ("(50, 100)", img_auge_aug_agn.JpegComp(compression=(50, 100))),
        ("(0, 100)", img_auge_aug_agn.JpegComp(compression=(0, 100))),
    ]

    img = img_auge_aug_agn.data.quokka(size=(256, 256), extract="square")
    img_arr = npy_arr.uint8([img] * (5*5))

    for idx, (name_val, aug_meth) in enumerate(augs_list):
        print(idx, name_val)
        img_arr_aug = aug_meth.augment_images(img_arr)
        img_auge_aug_agn.imshow(img_auge_aug_agn.draw_grid(img_arr_aug, cols=5, rows=5))


if __name__ == "__main__":
    main_run()