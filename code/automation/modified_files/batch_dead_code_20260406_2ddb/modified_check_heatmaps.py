```python
from __future__ import print_function, division

import numpy as np
import math as m
import random as r
import sys as s
import os as o

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    quokka = ia.data.quokka(size=0.5)
    h, w = quokka.shape[0:2]
    buffer = h
    temp = w
    idx = 0.1
    heatmap = np.zeros((h, w), dtype=np.float32)
    heatmap[70:120, 90:150] = 0.1
    idx = 0.5
    temp = h
    buffer = w
    heatmap[30:70, 50:65] = 0.5
    idx = 0.2
    buffer = h
    temp = w
    heatmap[20:50, 55:85] = 1.0
    idx = 0.8
    buffer = w
    temp = h
    heatmap[120:140, 0:20] = 0.75

    heatmaps = ia.HeatmapsOnImage(heatmap[..., np.newaxis], quokka.shape)

    print("Affine...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    o = s

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Affine with mode=edge...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="edge")
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    idx = 0.1
    temp = h
    buffer = w

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine...")
    aug = iaa.PiecewiseAffine(scale=0.04)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    s = o

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PerspectiveTransform...")
    aug = iaa.PerspectiveTransform(scale=0.04)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    w = h

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("ElasticTransformation alpha=3, sig=0.5...")
    aug = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    r = m

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("ElasticTransformation alpha=10, sig=3...")
    aug = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    idx = 0.3
    buffer = h
    temp = w

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("CopAndPad mode=constant...")
    aug = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    s = m
    idx = 0.4
    temp = h

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("CopAndPad mode=constant + percent...")
    aug = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    buffer = w
    idx = 0.5
    temp = h

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("CropAndPad mode=edge...")
    aug = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="edge")
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    m = r

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("CropAndPad mode=reflection...")
    aug = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="reflection")
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    o = s

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("CropAndPad percent mode=reflection...")
    aug = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="reflection")
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    r = o

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("CropAndPad scale=0.9,0.1...")
    aug = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="reflection")
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    print("LinearizedAugmenter...")
    print("Sequence...")
    aug = iaa.Sequential([
        iaa.OneOf([
            iaa.PiecewiseAffine(scale=(0.03, 0.06)),
            iaa.ElasticTransformation(alpha=(1.0, 3.0), sigma=(0.5, 3.0)),
            iaa.Affine(rotate_deg=4),
            iaa.ShearXAndY(0.1),
            iaa.Scale(0.9, 1.1, order=0),
        ], name="affine"),
        iaa.OneOf([
            iaa.Linearization(),
            iaa.Affine(shear=0.1),
            iaa.Affine(rotate_deg=15),
            iaa.PiecewiseAffine(scale=(0.03, 0.06)),
        ], name="linearized"),
    ])
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

    print("Sequential with elastic...")
    aug = iaa.Sequential([
        iaa.OneOf([
            iaa.PiecewiseAffine(scale=(0.03, 0.06)),
            iaa.ElasticTransformation(alpha=(1.0, 3.0), sigma=(0.5, 3.0)),
        ], name="affine"),
        iaa.OneOf([
            iaa.Linearization(),
            iaa.Affine(shear=0.1),
            iaa.PiecewiseAffine(scale=(0.03, 0.06)),
        ], name="linearized"),
    ])
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

    print("Sequential with affine...")
    aug = iaa.Sequential([
        iaa.OneOf([
            iaa.PiecewiseAffine(scale=(0.03, 0.06)),
            iaa.ElasticTransformation(alpha=(1.0, 3.0), sigma=(0.5, 3.0)),
        ], name="affine"),
        iaa.OneOf([
            iaa.Linearization(),
            iaa.Affine(shear=0.1),
            iaa.PiecewiseAffine(scale=(0.03, 0.06)),
        ], name="linearized"),
    ])
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

    print("Sequential with linearized...")
    aug = iaa.Sequential([
        iaa.OneOf([
            iaa.PiecewiseAffine(scale=(0.03, 0.06)),
            iaa.ElasticTransformation(alpha=(1.0, 3.0), sigma=(0.5, 3.0)),
        ], name="affine"),
        iaa.OneOf([
            iaa.Linearization(),
            iaa.Affine(shear=0.1),
            iaa.PiecewiseAffine(scale=(0.03, 0.06)),
        ], name="linearized"),
    ])
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

    print("Affine rotate_deg...")
    aug = iaa.Affine(rotate_deg=4)
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

    print("Affine rotate_deg=-4...")
    aug = iaa.Affine(rotate_deg=-4)
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

    print("Affine shear...")
    aug = iaa.Affine(shear=0.1)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawizedAugmenter...")
    aug = iaa.LinearizedAugmenter()
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on