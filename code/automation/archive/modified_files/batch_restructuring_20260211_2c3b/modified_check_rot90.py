from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def process_augmentation_steps(augs, image, augmentation_type):
    print("--------")
    print(f"Image + {augmentation_type}")
    print("--------")
    for name, aug in augs:
        print(name, "...")
        aug_det = aug.to_deterministic()
        augmented_images = aug_det.augment_images([image] * 16)
        if augmentation_type == "Keypoints":
            augmented_data = aug_det.augment_keypoints([ia.quokka_keypoints(0.25)] * 16)
            combined_images = [
                kps.draw_on_image(img, size=5)
                for img, kps in zip(augmented_images, augmented_data)
            ]
        elif augmentation_type == "Heatmaps":
            augmented_data = aug_det.augment_heatmaps([ia.quokka_heatmap(0.10)] * 16)
            combined_images = [
                hm.draw_on_image(img)[0]
                for img, hm in zip(augmented_images, augmented_data)
            ]
        ia.imshow(ia.draw_grid(combined_images))


def main():
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
    original_image = ia.data.quokka(0.25)
    process_augmentation_steps(augs, original_image, "Keypoints")
    process_augmentation_steps(augs, original_image, "Heatmaps")


if __name__ == "__main__":
    main()