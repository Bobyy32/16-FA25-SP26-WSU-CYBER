from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def process():
    data = ia.data.quokka(size=0.5)
    h, w = data.shape[0:2]
    result = np.zeros((h, w), dtype=np.float32)
    result[70:120, 90:150] = 0.1
    result[30:70, 50:65] = 0.5
    result[20:50, 55:85] = 1.0
    result[120:140, 0:20] = 0.75

    item = ia.HeatmapsOnImage(result[..., np.newaxis], data.shape)

    print("Affine...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    data_aug = aug.augment_image(data)
    result_aug = aug.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )

    print("Affine with mode=edge...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="edge")
    data_aug = aug.augment_image(data)
    result_aug = aug.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine...")
    aug = iaa.PiecewiseAffine(scale=0.04)
    aug_det = aug.to_deterministic()
    data_aug = aug_det.augment_image(data)
    result_aug = aug_det.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )

    print("PerspectiveTransform...")
    aug = iaa.PerspectiveTransform(scale=0.04)
    aug_det = aug.to_deterministic()
    data_aug = aug_det.augment_image(data)
    result_aug = aug_det.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )

    print("ElasticTransformation alpha=3, sig=0.5...")
    aug = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
    aug_det = aug.to_deterministic()
    data_aug = aug_det.augment_image(data)
    result_aug = aug_det.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )

    print("ElasticTransformation alpha=10, sig=3...")
    aug = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    aug_det = aug.to_deterministic()
    data_aug = aug_det.augment_image(data)
    result_aug = aug_det.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )

    print("CopAndPad mode=constant...")
    aug = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
    aug_det = aug.to_deterministic()
    data_aug = aug_det.augment_image(data)
    result_aug = aug_det.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )

    print("CopAndPad mode=constant + percent...")
    aug = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128)
    aug_det = aug.to_deterministic()
    data_aug = aug_det.augment_image(data)
    result_aug = aug_det.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )

    print("CropAndPad mode=edge...")
    aug = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="edge")
    aug_det = aug.to_deterministic()
    data_aug = aug_det.augment_image(data)
    result_aug = aug_det.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )

    print("Resize...")
    aug = iaa.Resize(0.5, interpolation="nearest")
    aug_det = aug.to_deterministic()
    data_aug = aug_det.augment_image(data)
    result_aug = aug_det.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(ia.draw_grid([result_drawn[0], result_aug_drawn[0]], cols=2))

    print("Alpha...")
    aug = iaa.Alpha(0.7, iaa.Affine(rotate=20))
    aug_det = aug.to_deterministic()
    data_aug = aug_det.augment_image(data)
    result_aug = aug_det.augment_heatmaps([item])[0]
    result_drawn = item.draw_on_image(data)
    result_aug_drawn = result_aug.draw_on_image(data_aug)

    ia.imshow(
        np.hstack([
            result_drawn[0],
            result_aug_drawn[0]
        ])
    )


if __name__ == "__main__":
    process()