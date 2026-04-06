from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


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

sample_img = ia.data.quokka(0.25)

print("--------")
print("Image + Keypoints")
print("--------")
keypoints_arr = ia.quokka_keypoints(0.25)

for aug_config, deterministic_aug in augmentation_configs:
    print(aug_config, "...")
    augmented_images_list = deterministic_aug.augment_images([sample_img] * 16)
    augmented_keypoints = deterministic_aug.augment_keypoints([keypoints_arr] * 16)
    drawn_results = [kp.draw_on_image(img, size=5) for img, kp in zip(augmented_images_list, augmented_keypoints)]
    ia.imshow(ia.draw_grid(drawn_results))

print("--------")
print("Image + Heatmaps (low res)")
print("--------")
heatmap_arr = ia.quokka_heatmap(0.10)

for aug_config, deterministic_aug in augmentation_configs:
    print(aug_config, "...")
    augmented_images_list = deterministic_aug.augment_images([sample_img] * 16)
    augmented_heatmaps = deterministic_aug.augment_heatmaps([heatmap_arr] * 16)
    drawn_results = [hm.draw_on_image(img)[0] for img, hm in zip(augmented_images_list, augmented_heatmaps)]
    ia.imshow(ia.draw_grid(drawn_results))


if __name__ == "__main__":
    main()