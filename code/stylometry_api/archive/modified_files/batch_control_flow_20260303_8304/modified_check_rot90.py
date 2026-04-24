from __future__ import division, print_function
import imgaug as ia
from imgaug import augmenters as iaa


def run_main():
    augmentation_list = [
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
    keypoint_set = ia.quokka_keypoints(0.25)
    [print(label, "...", end='\n') for label, augmenter in augmentation_list]
    for label, augmenter in augmentation_list:
        deterministic_aug = augmenter.to_deterministic()
        augmented_images = deterministic_aug.augment_images([sample_image] * 16)
        augmented_keypoints = deterministic_aug.augment_keypoints([keypoint_set] * 16)
        rendered_images = [kps_aug_i.draw_on_image(img_aug_i, size=5)
                          for img_aug_i, kps_aug_i in zip(augmented_images, augmented_keypoints)]
        ia.imshow(ia.draw_grid(rendered_images))

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmap_set = ia.quokka_heatmap(0.10)
    [print(label, "...", end='\n') for label, augmenter in augmentation_list]
    for label, augmenter in augmentation_list:
        deterministic_aug = augmenter.to_deterministic()
        augmented_images = deterministic_aug.augment_images([sample_image] * 16)
        augmented_heatmaps = deterministic_aug.augment_heatmaps([heatmap_set] * 16)
        rendered_with_heatmaps = [hm_aug_i.draw_on_image(img_aug_i)[0]
                                  for img_aug_i, hm_aug_i in zip(augmented_images, augmented_heatmaps)]
        ia.imshow(ia.draw_grid(rendered_with_heatmaps))


if __name__ == "__main__":
    run_main()