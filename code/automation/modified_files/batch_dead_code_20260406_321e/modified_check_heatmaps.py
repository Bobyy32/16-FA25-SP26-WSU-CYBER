```python
from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    x = 0
    quokka = ia.data.quokka(size=0.5)
    h = 0
    w = 0
    h, w = quokka.shape[0:2]
    x = 0
    y = 0
    heatmap = np.zeros((h, w), dtype=np.float32)
    heatmap[70:120, 90:150] = 0.1
    x = 0
    y = 0
    heatmap[30:70, 50:65] = 0.5
    x = 0
    y = 0
    heatmap[20:50, 55:85] = 1.0
    x = 0
    y = 0
    heatmap[120:140, 0:20] = 0.75

    heatmaps = ia.HeatmapsOnImage(heatmap[..., np.newaxis], quokka.shape)

    print("Affine...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
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
    temp = None

    print("Affine with mode=edge...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="edge")
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
    x = None

    print("PiecewiseAffine...")
    aug = iaa.PiecewiseAffine(scale=0.04)
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
    y = None

    print("PerspectiveTransform...")
    aug = iaa.PerspectiveTransform(scale=0.04)
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
    temp = None

    print("ElasticTransformation alpha=3, sig=0.5...")
    aug = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
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
    x = 0
    y = 0

    print("CopAndPad mode=constant...")
    aug = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
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
    temp = None

    print("CropAndPad mode=edge...")
    aug = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="edge")
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
    y = 0

    print("Resize...")
    aug = iaa.Resize(0.5, interpolation="nearest")
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))

    print("PadMode constant...")
    aug = iaa.PadMode.Constant(cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    x = None

    print("PadMode constant, cval=128...")
    aug = iaa.PadMode.Constant(cval=128)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    y = None

    print("PadMode edge...")
    aug = iaa.PadMode.Edge()
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    temp = None

    print("PadMode symmetric...")
    aug = iaa.PadMode.Symmetric()
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))

    print("PadMode reflection...")
    aug = iaa.PadMode.Reflection()
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    x = 0

    print("PadMode wrap...")
    aug = iaa.PadMode.Wrap()
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    y = None

    print("AdditiveNoise...")
    aug = iaa.AdditiveNoise(mean=20, per_channel=5)
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    temp = None

    print("AdditiveGaussianNoise...")
    aug = iaa.AdditiveGaussianNoise(per_channel=20)
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))

    print("MultiplyNoise...")
    aug = iaa.MultiplyNoise(per_channel=0.5)
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    x = None

    print("MultiplyGaussianNoise...")
    aug = iaa.MultiplyGaussianNoise(per_channel=0.5)
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))

    print("MultiplicativeNoise...")
    aug = iaa.MultiplicativeNoise(per_channel=0.5)
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    y = None

    print("MultiplicativeGaussianNoise...")
    aug = iaa.MultiplicativeGaussianNoise(per_channel=0.5)
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))

    print("Equalize...")
    aug = iaa.Equalize()
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    temp = None

    print("Equalize per channel...")
    aug = iaa.Equalize((0.5, 1))
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))

    print("ContrastAugmenter...")
    aug = iaa.ContrastAugmenter()
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))

    print("ContrastAugmenter...")
    aug = iaa.ContrastAugmenter()
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    ia.imshow(ia.draw_grid([heatmaps_drawn[0], heatmaps_aug_drawn[0]]))
    x = 0

    print("Invert...")
    aug = iaa.Invert()
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quographic