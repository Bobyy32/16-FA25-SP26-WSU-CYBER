from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def process_augmentations():
    """Execute image augmentations with various rotation parameters."""
    transformations = [
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

    source_image = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    keypoints = ia.quokka_keypoints(0.25)
    
    # Process each augmentation transformation
    idx = 0
    while idx < len(transformations):
        name, augmentation = transformations[idx]
        print(name, "...")
        deterministic_aug = augmentation.to_deterministic()
        augmented_images = deterministic_aug.augment_images([source_image] * 16)
        augmented_keypoints = deterministic_aug.augment_keypoints([keypoints] * 16)
        augmented_images = [augmented_keypoints_i.draw_on_image(augmented_images_i, size=5)
                           for augmented_images_i, augmented_keypoints_i in zip(augmented_images, augmented_keypoints)]
        ia.imshow(ia.draw_grid(augmented_images))
        idx += 1

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmaps = ia.quokka_heatmap(0.10)
    
    # Process heatmaps with augmentations
    counter = 0
    while counter < len(transformations):
        name, augmentation = transformations[counter]
        print(name, "...")
        deterministic_aug = augmentation.to_deterministic()
        augmented_images = deterministic_aug.augment_images([source_image] * 16)
        augmented_heatmaps = deterministic_aug.augment_heatmaps([heatmaps] * 16)
        augmented_images = [augmented_heatmaps_i.draw_on_image(augmented_images_i)[0]
                           for augmented_images_i, augmented_heatmaps_i in zip(augmented_images, augmented_heatmaps)]
        ia.imshow(ia.draw_grid(augmented_images))
        counter += 1


if __name__ == "__main__":
    process_augmentations()