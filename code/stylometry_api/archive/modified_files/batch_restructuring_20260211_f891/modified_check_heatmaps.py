from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = ia.data.quokka(size=0.5)
    height, width = image.shape[0:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    heatmap[70:120, 90:150] = 0.1
    heatmap[30:70, 50:65] = 0.5
    heatmap[20:50, 55:85] = 1.0
    heatmap[120:140, 0:20] = 0.75

    heatmaps = ia.HeatmapsOnImage(heatmap[..., np.newaxis], image.shape)

    augment_and_visualize("Affine", iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128), image, heatmaps)
    augment_and_visualize("Affine with mode=edge", iaa.Affine(translate_px={"x": 20}, mode="edge"), image, heatmaps)
    augment_and_visualize("PiecewiseAffine", iaa.PiecewiseAffine(scale=0.04), image, heatmaps)
    augment_and_visualize("PerspectiveTransform", iaa.PerspectiveTransform(scale=0.04), image, heatmaps)
    augment_and_visualize("ElasticTransformation alpha=3, sig=0.5", iaa.ElasticTransformation(alpha=3.0, sigma=0.5), image, heatmaps)
    augment_and_visualize("ElasticTransformation alpha=10, sig=3", iaa.ElasticTransformation(alpha=10.0, sigma=3.0), image, heatmaps)
    augment_and_visualize("CopAndPad mode=constant", iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128), image, heatmaps)
    augment_and_visualize("CopAndPad mode=constant + percent", iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128), image, heatmaps)
    augment_and_visualize("CropAndPad mode=edge", iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="edge"), image, heatmaps)
    augment_and_visualize("Resize", iaa.Resize(0.5, interpolation="nearest"), image, heatmaps)
    augment_and_visualize("Alpha", iaa.Alpha(0.7, iaa.Affine(rotate=20)), image, heatmaps)


def augment_and_visualize(name, augmenter, image, heatmaps):
    print(name + "...")
    aug_det = augmenter.to_deterministic()
    image_aug = aug_det.augment_image(image)
    heatmaps_aug = aug_det.augment_heatmaps([heatmaps])[0]
    heatmaps_drawn = heatmaps.draw_on_image(image)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(image_aug)

    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )


if __name__ == "__main__":
    main()