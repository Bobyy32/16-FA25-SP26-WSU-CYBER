from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def execution_flow():
    transformation_list = [
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

    sample_image = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    keypoints = ia.quokka_keypoints(0.25)
    
    index = 0
    while index < len(transformation_list):
        name, augmentation = transformation_list[index]
        print(name, "...")
        deterministic_aug = augmentation.to_deterministic()
        augmented_images = deterministic_aug.augment_images([sample_image] * 16)
        augmented_keypoints = deterministic_aug.augment_keypoints([keypoints] * 16)
        augmented_images = [augmented_keypoints_i.draw_on_image(augmented_images_i, size=5)
                            for augmented_images_i, augmented_keypoints_i in zip(augmented_images, augmented_keypoints)]
        ia.imshow(ia.draw_grid(augmented_images))
        index += 1

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmaps = ia.quokka_heatmap(0.10)
    
    iterator = 0
    while iterator < len(transformation_list):
        name, augmentation = transformation_list[iterator]
        print(name, "...")
        deterministic_aug = augmentation.to_deterministic()
        augmented_images = deterministic_aug.augment_images([sample_image] * 16)
        augmented_heatmaps = deterministic_aug.augment_heatmaps([heatmaps] * 16)
        augmented_images = [augmented_heatmaps_i.draw_on_image(augmented_images_i)[0]
                            for augmented_images_i, augmented_heatmaps_i in zip(augmented_images, augmented_heatmaps)]
        ia.imshow(ia.draw_grid(augmented_images))
        iterator += 1


if __name__ == "__main__":
    execution_flow()