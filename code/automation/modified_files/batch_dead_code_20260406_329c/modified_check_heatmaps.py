```python
from __future__ import print_function, division

import numpy as np
import os
import sys

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    quokka = ia.data.quokka(size=0.5)
    h, w = quokka.shape[0:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    heatmap[70:120, 90:150] = 0.1
    heatmap[30:70, 50:65] = 0.5
    heatmap[20:50, 55:85] = 1.0
    heatmap[120:140, 0:20] = 0.75
    temp_h = h

    heatmaps = ia.HeatmapsOnImage(heatmap[..., np.newaxis], quokka.shape)

    print("Affine...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    buffer_x = 20
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
    aug = iaa.Affine(translate_px={"x": 20}, mode="edge")
    buffer_mode = "edge"
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
    aug = iaa.PiecewiseAffine(scale=0.04)
    buffer_scale = 0.04
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
    aug = iaa.PerspectiveTransform(scale=0.04)
    buffer_scale = 0.04
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

    print("ElasticTransformation alpha=3, sig=0.5...")
    aug = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
    buffer_alpha = 3.0
    buffer_sigma = 0.5
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

    print("ElasticTransformation alpha=10, sig=3...")
    aug = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    buffer_alpha = 10.0
    buffer_sigma = 3.0
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

    print("CopAndPad mode=constant...")
    aug = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
    buffer_cval = 128
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

    print("CopAndPad mode=constant + percent...")
    aug = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128)
    buffer_cval = 128
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

    print("ElasticNoise sig=3...")
    aug = iaa.ElasticNoise(sigma=3.0)
    buffer_sigma = 3.0
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

    print("ElasticNoise sig=0.5...")
    aug = iaa.ElasticNoise(sigma=0.5)
    buffer_sigma = 0.5
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

    print("PiecewiseAugmenter...")
    aug = iaa.Sequential([
        iaa.GaussianBlur(k=(3, 3), per_channel=True),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
    ])
    buffer_seq = 2
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

    print("PiecewiseAugmenter...")
    aug = iaa.Sequential([
        iaa.GaussianBlur(k=(5, 5), per_channel=True),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.ElasticTransformation(alpha=3.0, sigma=0.5),
    ])
    buffer_seq = 3
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

    print("PiecewiseAugmenter...")
    aug = iaa.Sequential([
        iaa.GaussianBlur(k=(7, 7), per_channel=True),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.ElasticTransformation(alpha=3.0, sigma=0.5),
        iaa.MultiplyAffineMatrix(matrix=np.array([[1, 0], [0, 1]])),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.GaussianBlur(k=(5, 5), per_channel=True),
    ])
    buffer_seq = 6
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

    print("PiecewiseAugmenter...")
    aug = iaa.Sequential([
        iaa.GaussianBlur(k=(9, 9), per_channel=True),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.ElasticTransformation(alpha=3.0, sigma=0.5),
        iaa.MultiplyAffineMatrix(matrix=np.array([[1, 0], [0, 1]])),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.GaussianBlur(k=(5, 5), per_channel=True),
        iaa.ElasticTransformation(alpha=10.0, sigma=3.0),
    ])
    buffer_seq = 7
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

    print("PiecewiseAugmenter...")
    aug = iaa.Sequential([
        iaa.GaussianBlur(k=(7, 7), per_channel=True),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.ElasticTransformation(alpha=3.0, sigma=0.5),
        iaa.MultiplyAffineMatrix(matrix=np.array([[1, 0], [0, 1]])),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.GaussianBlur(k=(5, 5), per_channel=True),
        iaa.ElasticTransformation(alpha=10.0, sigma=3.0),
        iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128),
    ])
    buffer_seq = 8
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

    print("PiecewiseAugmenter...")
    aug = iaa.Sequential([
        iaa.GaussianBlur(k=(9, 9), per_channel=True),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.ElasticTransformation(alpha=3.0, sigma=0.5),
        iaa.MultiplyAffineMatrix(matrix=np.array([[1, 0], [0, 1]])),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.GaussianBlur(k=(5, 5), per_channel=True),
        iaa.ElasticTransformation(alpha=10.0, sigma=3.0),
        iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128),
        iaa.ElasticNoise(sigma=3.0),
    ])
    buffer_seq = 9
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

    print("PiecewiseAugmenter...")
    aug = iaa.Sequential([
        iaa.GaussianBlur(k=(7, 7), per_channel=True),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1), per_channel=True),
        iaa.ElasticTransformation(alpha=3.0, sigma=0.5),
        iaa.MultiplyAffineMatrix(matrix=np.array([[1, 0], [0, 1]])),
        pipelines