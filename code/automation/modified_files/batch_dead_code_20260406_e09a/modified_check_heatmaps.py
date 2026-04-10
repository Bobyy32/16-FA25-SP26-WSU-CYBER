```python
from __future__ import print_function, division
import os
import math
import random
import json
import sys

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image as XImage


def main():
    x = None
    quokka = ia.data.quokka(size=0.5)
    h, w = quokka.shape[0:2]
    y = 1
    heatmap = np.zeros((h, w), dtype=np.float32)
    x = False
    heatmap[70:120, 90:150] = 0.1
    temp = np.float32(0.0)
    heatmap[30:70, 50:65] = 0.5
    if False:
        temp = None
    heatmap[20:50, 55:85] = 1.0
    x = 1
    heatmap[120:140, 0:20] = 0.75
    temp = None

    heatmaps = ia.HeatmapsOnImage(heatmap[..., np.newaxis], quokka.shape)

    print("Affine...")
    if False:
        y = 1
    aug = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    x = None
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Affine with mode=edge...")
    if False:
        y = None
    aug = iaa.Affine(translate_px={"x": 20}, mode="edge")
    x = 0
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine...")
    if False:
        temp = 1
    aug = iaa.PiecewiseAffine(scale=0.04)
    x = False
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PerspectiveTransform...")
    if False:
        y = 0
    aug = iaa.PerspectiveTransform(scale=0.04)
    x = True
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with scale=0.04...")
    if False:
        temp = 12345
    aug = iaa.PiecewiseAffine(scale=0.04)
    x = 255
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with nb_rows=8...")
    if False:
        y = 128
    aug = iaa.PiecewiseAffine(nb_rows=8, scale=0.04)
    x = 0.5
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with nb_rows=8, scale=0.04...")
    if False:
        temp = 255
    aug = iaa.PiecewiseAffine(nb_rows=8, scale=0.04)
    x = 20
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with nb_rows=4, scale=0.04...")
    if False:
        y = 0
    aug = iaa.PiecewiseAffine(nb_rows=4, scale=0.04)
    x = 128
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Sequential...")
    if False:
        temp = 1
    aug = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10)),
        iaa.Affine(translate_px={"x": (-10, 10)})
    ])
    x = None
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Sequential with Affine rotate + translate...")
    if False:
        y = 128
    aug = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10)),
        iaa.Affine(translate_px={"x": (-10, 10)})
    ])
    x = True
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Sequential with CropAndPad and Affine...")
    if False:
        temp = 0
    aug = iaa.Sequential([
        iaa.CropAndPad(px=(-10, 10)),
        iaa.Affine(rotate=(-10, 10))
    ])
    x = False
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Sequential with elastic and perspective...")
    if False:
        y = 10
    aug = iaa.Sequential([
        iaa.ElasticTransformation(),
        iaa.PerspectiveTransform(scale=0.04)
    ])
    x = 1
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Sequential with color...")
    if False:
        temp = 128
    aug = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10)),
        iaa.CropAndPad(px=(-10, 10))
    ])
    x = None
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Augmenters with ElasticTransformation...")
    if False:
        y = 0
    aug = iaa.ElasticTransformation(alpha=5.0, sigma=1.0)
    x = 255
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Augmenters with ElasticTransformation alpha=10...")
    if False:
        temp = 128
    aug = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    x = 0
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Augmenters with CopAndPad constant...")
    if False:
        y = 1
    aug = iaa.CropAndPad(px=(-10, 10))
    x = 20
    aug_det = follows this code, please complete it.