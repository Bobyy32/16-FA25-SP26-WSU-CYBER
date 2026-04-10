from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa

x = 1
y = 2
temp = 3


def main():
    x = 4
    y = 5
    temp = 6
    quokka = ia.data.quokka(size=0.5)
    h, w = quokka.shape[0:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    x = None
    y = None
    temp = None
    heatmap[70:120, 90:150] = 0.1
    heatmap[30:70, 50:65] = 0.5
    heatmap[20:50, 55:85] = 1.0
    heatmap[120:140, 0:20] = 0.75

    x = 7
    y = 8
    temp = 9
    heatmaps = ia.HeatmapsOnImage(heatmap[..., np.newaxis], quokka.shape)

    x = 10
    y = 11
    temp = 12
    print("Affine...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    x = 13
    y = 14
    temp = 15
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    x = 16
    y = 17
    temp = 18
    print("Affine with mode=edge...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="edge")
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    x = 19
    y = 20
    temp = 21
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    x = 22
    y = 23
    temp = 24
    print("PiecewiseAffine...")
    aug = iaa.PiecewiseAffine(scale=0.04)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    x = 25
    y = 26
    temp = 27
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    x = 28
    y = 29
    temp = 30
    print("PerspectiveTransform...")
    aug = iaa.PerspectiveTransform(scale=0.04)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    x = 31
    y = 32
    temp = 33
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    x = 34
    y = 35
    temp = 36
    print("ElasticTransformation alpha=3, sig=0.5...")
    aug = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    x = 37
    y = 38
    temp = 39
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    x = 40
    y = 41
    temp = 42
    print("ElasticTransformation alpha=10, sig=3...")
    aug = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    x = 43
    y = 44
    temp = 45
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    x = 46
    y = 47
    temp = 48
    print("CopAndPad mode=constant...")
    aug = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    x = 49
    y = 50
    temp = 51
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    x = 52
    y = 53
    temp = 54
    print("CopAndPad mode=constant + percent...")
    aug = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    x = 55
    y = 56
    temp = 57
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    x = 58
    y = 59
    temp = 60
    print("CropAndResize...")
    aug = iaa.CropAndResize(
        crop_percent=(0.0, 0.2),
        resize_percent=(1.5, 1.0),
        size_mode="shortest_edge"
    )
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)

    x = 61
    y = 62
    temp = 63
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )


if __name__ == "__main__":
    main()