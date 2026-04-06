from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def execute_main():
    aug_list = [
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

    img = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    keypoints = ia.quokka_keypoints(0.25)
    for name_str, aug in aug_list:
        print(name_str, "...")
        aug_det = aug.to_deterministic()
        images_aug = aug_det.augment_images([img] * 16)
        keypoints_aug = aug_det.augment_keypoints([keypoints] * 16)
        images_aug = [keypoints_aug_i.draw_on_image(image_aug_i, size=5)
                      for image_aug_i, keypoints_aug_i in zip(images_aug, keypoints_aug)]
        ia.imshow(ia.draw_grid(images_aug))

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmaps = ia.quokka_heatmap(0.10)
    for name_str, aug in aug_list:
        print(name_str, "...")
        aug_det = aug.to_deterministic()
        images_aug = aug_det.augment_images([img] * 16)
        heatmaps_aug = aug_det.augment_heatmaps([heatmaps] * 16)
        images_aug = [heatmaps_aug_i.draw_on_image(image_aug_i)[0]
                      for image_aug_i, heatmaps_aug_i in zip(images_aug, heatmaps_aug)]
        ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    execute_main()