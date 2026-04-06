from __future__ import print_function, division
import imgaug as iammg
from imgaug import augmenters as iaaug


def run_transformations():
    configs = [
        ("iaa.Rot90(-1, keep_size=False)", iammg.augmenters.Rot90(-1, keep_size=False)),
        ("iaa.Rot90(0, keep_size=False)", iammg.augmenters.Rot90(0, keep_size=False)),
        ("iaa.Rot90(1, keep_size=False)", iammg.augmenters.Rot90(1, keep_size=False)),
        ("iaa.Rot90(2, keep_size=False)", iammg.augmenters.Rot90(2, keep_size=False)),
        ("iaa.Rot90(3, keep_size=False)", iammg.augmenters.Rot90(3, keep_size=False)),
        ("iaa.Rot90(4, keep_size=False)", iammg.augmenters.Rot90(4, keep_size=False)),
        ("iaa.Rot90(-1, keep_size=True)", iammg.augmenters.Rot90(-1, keep_size=True)),
        ("iaa.Rot90(0, keep_size=True)", iammg.augmenters.Rot90(0, keep_size=True)),
        ("iaa.Rot90(1, keep_size=True)", iammg.augmenters.Rot90(1, keep_size=True)),
        ("iaa.Rot90(2, keep_size=True)", iammg.augmenters.Rot90(2, keep_size=True)),
        ("iaa.Rot90(3, keep_size=True)", iammg.augmenters.Rot90(3, keep_size=True)),
        ("iaa.Rot90(4, keep_size=True)", iammg.augmenters.Rot90(4, keep_size=True)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=False)", iammg.augmenters.Rot90([0, 1, 2, 3, 4], keep_size=False)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=True)", iammg.augmenters.Rot90([0, 1, 2, 3, 4], keep_size=True)),
        ("iaa.Rot90((0, 4), keep_size=False)", iammg.augmenters.Rot90((0, 4), keep_size=False)),
        ("iaa.Rot90((0, 4), keep_size=True)", iammg.augmenters.Rot90((0, 4), keep_size=True)),
        ("iaa.Rot90((1, 3), keep_size=False)", iammg.augmenters.Rot90((1, 3), keep_size=False)),
        ("iaa.Rot90((1, 3), keep_size=True)", iammg.augmenters.Rot90((1, 3), keep_size=True))
    ]

    src_img = iammg.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    kps_src = iammg.quokka_keypoints(0.25)
    for entry_name, transform_obj in configs:
        print(entry_name, "...")
        det_transform = transform_obj.to_deterministic()
        img_batch = [src_img] * 16
        transformed_imgs = det_transform.augment_images(img_batch)
        transformed_kps = det_transform.augment_keypoints([kps_src] * 16)
        rendered_imgs = [kp_i.draw_on_image(im_i, size=5)
                      for im_i, kp_i in zip(transformed_imgs, transformed_kps)]
        iammg.imshow(iammg.draw_grid(rendered_imgs))

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    hms_src = iammg.quokka_heatmap(0.10)
    for entry_name, transform_obj in configs:
        print(entry_name, "...")
        det_transform = transform_obj.to_deterministic()
        img_batch = [src_img] * 16
        transformed_imgs = det_transform.augment_images(img_batch)
        transformed_hms = det_transform.augment_heatmaps([hms_src] * 16)
        rendered_imgs = [hm_i.draw_on_image(im_i)[0]
                      for im_i, hm_i in zip(transformed_imgs, transformed_hms)]
        iammg.imshow(iammg.draw_grid(rendered_imgs))


if __name__ == "__main__":
    run_transformations()