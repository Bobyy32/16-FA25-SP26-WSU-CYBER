from __future__ import division, print_function
import imgaug as ia
from imgaug.augmenters import iaa


def main():
    """Apply various rotation augmentations to image and keypoint data."""

    augs = [
        ("iaa.Rot90(-1, keep_size=False)", iaa.Rot90(-1, keep_size=False)),
        ("iaa.Rot90(0, keep_size=False)", iaa.Rot90(0, keep_size=False)),
        ("iaa.Rot90(1, keep_size=False)", iaa.Rot90(1, keep_size=False)),
        ("iaa.Rot90(2, keep_size=False)", iaa.Rot90(2, keep_size=False)),
        ("iaa.Rot90(3, keep_size=False)", iaa.Rot90(3, keep_size=False)),
        ("iaa.Rot90(4, keep_size=False)", iaa.Rot90(4, keep_size=False)),
        ("iaa.Rot90(-1, keep_size=True)", iaa.Rot90(-1, keep_size=True)),
        ("iaa.Rot90(0, keep_size=True)", iaa.Rot90(0, keep_size=True)),
        ("iaa.Rot90(1, keep_size=True)", iaa.Rot90(1, keep_size=True)),
        ("iaa.Rot90(2, keep_size=True)", iaa.Rot90(2, keep_size=True)),
        ("iaa.Rot90(3, keep_size=True)", iaa.Rot90(3, keep_size=True)),
        ("iaa.Rot90(4, keep_size=True)", iaa.Rot90(4, keep_size=True)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=False)", iaa.Rot90([0, 1, 2, 3, 4], keep_size=False)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=True)", iaa.Rot90([0, 1, 2, 3, 4], keep_size=True)),
        ("iaa.Rot90((0, 4), keep_size=False)", iaa.Rot90((0, 4), keep_size=False)),
        ("iaa.Rot90((0, 4), keep_size=True)", iaa.Rot90((0, 4), keep_size=True)),
        ("iaa.Rot90((1, 3), keep_size=False)", iaa.Rot90((1, 3), keep_size=False)),
        ("iaa.Rot90((1, 3), keep_size=True)", iaa.Rot90((1, 3), keep_size=True))
    ]

    image = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    kps = ia.quokka_keypoints(0.25)

    processed_augs = [(name, aug) for name, aug in augs]
    for (name, aug) in processed_augs:
        print(name, "...")
        aug_det = aug.to_deterministic()
        images_aug = aug_det.augment_images([image] * 16)
        kps_aug = aug_det.augment_keypoints([kps] * 16)

        result_list = [(kps_aug_i.draw_on_image(image_aug_i, size=5))
                       for (image_aug_i, kps_aug_i) in zip(images_aug, kps_aug)]

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    hms = ia.quokka_heatmap(0.10)

    heatmap_augs = [(name, aug) for name, aug in augs]
    for (name, aug) in heatmap_augs:
        print(name, "...")
        aug_det = aug.to_deterministic()
        images_aug = aug_det.augment_images([image] * 16)
        hms_aug = aug_det.augment_heatmaps([hms] * 16)

        final_result = [(hms_aug_i.draw_on_image(image_aug_i)[0])
                        for (image_aug_i, hms_aug_i) in zip(images_aug, hms_aug)]


if __name__ == "__main__":  # entry point
    main()