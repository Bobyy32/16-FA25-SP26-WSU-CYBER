from __future__ import division, print_function
import imgaug as ia
from imgaug import augmenters as iaa


def execute_main_flow():
    augmentation_configs = [
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

    target_image = ia.data.quokka(0.25)

    # Display image rotation variations with keypoint tracking
    print("--------")
    print("Image + Keypoints")
    print("--------")
    keypoints_dataset = ia.quokka_keypoints(0.25)
    for label, augment in augmentation_configs:
        print(label, "...")
        augmented = augment.to_deterministic()
        transformed_images = augmented.augment_images([target_image] * 16)
        transformed_keypoints = augmented.augment_keypoints([keypoints_dataset] * 16)
        transformed_images = [kp_i.draw_on_image(img_i, size=5)
                            for img_i, kp_i in zip(transformed_images, transformed_keypoints)]
        ia.imshow(ia.draw_grid(transformed_images))

    # Display heatmaps with rotation variations
    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmap_dataset = ia.quokka_heatmap(0.10)
    for label, augment in augmentation_configs:
        print(label, "...")
        augmented = augment.to_deterministic()
        transformed_images = augmented.augment_images([target_image] * 16)
        transformed_heatmaps = augmented.augment_heatmaps([heatmap_dataset] * 16)
        transformed_images = [hm_i.draw_on_image(img_i)[0]
                            for img_i, hm_i in zip(transformed_images, transformed_heatmaps)]
        ia.imshow(ia.draw_grid(transformed_images))


if __name__ == "__main__":
    execute_main_flow()