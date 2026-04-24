```python
from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def _unused_helper():
    pass


def main():
    quokka = ia.data.quokka(size=0.5)
    h, w = quokka.shape[0:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    heatmap[70:120, 90:150] = 0.1
    heatmap[30:70, 50:65] = 0.5
    heatmap[20:50, 55:85] = 1.0
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
    _tmp = None

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
    _tmp = None

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
    _tmp = None

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
    _tmp = None

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
    _tmp = None

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
    _tmp = None

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
    _tmp = None

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
    _tmp = None

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
    _tmp = None

    print("Resize...")
    aug = iaa.Resize(0.5, interpolation="nearest")
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
    _tmp = None

    print("Flatten...")
    aug = iaa.Flipud(p=0.5)
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
    _tmp = None

    print("Flip...")
    aug = iaa.Fliplr(p=0.5)
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
    _tmp = None

    print("Rotate...")
    aug = iaa.AverageColorJitter(brightness=(0.1, 0.2))
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
    _tmp = None

    print("Rotate90...")
    aug = iaa.Rotate90(p=0.5)
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
    _tmp = None

    print("Rotate90deg...")
    aug = iaa.Rotate90Deg(p=0.5)
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
    _tmp = None

    print("Rotate90Degs...")
    aug = iaa.Rotate90Degs(p=0.5)
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
    _tmp = None

    print("Scale...")
    aug = iaa.Scale((0.7, 0.9), scale_mode="affine")
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
    _tmp = None

    print("ScaleAndPad...")
    aug = iaa.ScaleAndPad(scale=(0.7, 0.9), mode="constant", cval=128)
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
    _tmp = None

    print("ScaleAndPad with percent and mode...")
    aug = iaa.ScaleAndPad(scale=(0.7, 0.9), mode="constant", cval=128, scale_percent=(0.8, 1.0))
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
    _tmp = None

    print("Scale with interpolation...")
    aug = iaa.Scale(scale=0.5, interpolation="nearest")
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
    _tmp = None

    print("ShiftScaleGridWithFill...")
    aug = iaa.ShiftScaleGridWithFill(xshift_px=15, yshift_px=10, scale=0.05)
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
    _tmp = None

    print("ShiftScaleGridWithFill with percent...")
    aug = iaa.ShiftScaleGridWithFill(xshift_px=15, yshift_px=10, scale=0.05, scale_percent=(0.7, 1.3))
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
    _tmp = None

    print("ShiftScaleGridWithFill and CropAndPad...")
    aug = iaa.ShiftScaleGridWithFill(xshift_px=15, yshift_px=10, scale=0.05, crop=(0, 0, 10, 20))
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
    function
function