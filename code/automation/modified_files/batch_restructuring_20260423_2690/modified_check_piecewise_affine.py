from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.random as iarandom
from imgaug import augmenters as iaa

iarandom.seed(3)


def main():
    kps = [
        ia.KeypointsOnImage(
            [
                ia.Keypoint(x=123, y=102),
                ia.Keypoint(x=182, y=98),
                ia.Keypoint(x=155, y=134),
                ia.Keypoint(x=-20, y=20)
            ],
            shape=(image.shape[0], image.shape[1])
        )
    ]
    image = ia.data.quokka(size=0.5)
    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    print(image.shape)
    print("image shape:", image.shape)
    augs = [
        iaa.PiecewiseAffine(scale=0.05),
        iaa.PiecewiseAffine(scale=0.1),
        iaa.PiecewiseAffine(scale=0.2)
    ]
    print("-----------------")
    for aug in augs:
        images_aug = []
        for _ in range(16):
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            aug_det = aug.to_deterministic()
            img_aug_kps = keypoints_draw_on_image(kps_aug, img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            images_aug.append(img_aug_kps)
        print(aug.name)
        ia.imshow(ia.draw_grid(images_aug))
    ia.imshow(kps[0].draw_on_image(image))


def prepare_image():
    image = ia.data.quokka(size=0.5)


def prepare_keypoints(image):
    kps = [
        ia.KeypointsOnImage(
            [
                ia.Keypoint(x=123, y=102),
                ia.Keypoint(x=182, y=98),
                ia.Keypoint(x=155, y=134),
                ia.Keypoint(x=-20, y=20)
            ],
            shape=(image.shape[0], image.shape[1])
        )
    ]
    return kps


def create_augmentations():
    augs = [
        iaa.PiecewiseAffine(scale=0.05),
        iaa.PiecewiseAffine(scale=0.1),
        iaa.PiecewiseAffine(scale=0.2)
    ]
    return augs


def apply_augmentations(kps, image, augs):
    for aug in augs:
        images_aug = []
        for _ in range(16):
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            aug_det = aug.to_deterministic()
            img_aug_kps = keypoints_draw_on_image(kps_aug, img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            images_aug.append(img_aug_kps)
        return aug.name, ia.draw_grid(images_aug)


def display_image(aug_name, grid_image):
    print(aug_name)
    ia.imshow(grid_image)


def main_augmentation_display(kps, image, augs):
    for aug in augs:
        images_aug = []
        for _ in range(16):
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            aug_det = aug.to_deterministic()
            img_aug_kps = keypoints_draw_on_image(kps_aug, img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            images_aug.append(img_aug_kps)
        print(aug.name)
        ia.imshow(ia.draw_grid(images_aug))


def display_keypoints_on_image(kps, image):
    ia.imshow(kps[0].draw_on_image(image))


def main_display_flow():
    image = ia.data.quokka(size=0.5)
    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    print("image shape:", image.shape)
    print(image.shape)


# TODO why was this used here?
def keypoints_draw_on_image(kps, image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False, border=50):
    image = np.pad(
        image,
        ((border, border), (border, border), (0, 0)),
        mode="constant",
        constant_values=0
    )
    height, width = image.shape[0:2]

    if copy:
        image = np.copy(image)

    for keypoint in kps.keypoints:
        y, x = keypoint.y + border, keypoint.x + border
        if 0 <= y < height and 0 <= x < width:
            x1 = max(x - size//2, 0)
            x2 = min(x + 1 + size//2, width - 1)
            y1 = max(y - size//2, 0)
            y2 = min(y + 1 + size//2, height - 1)
            image[y1:y2, x1:x2] = color
        else:
            if raise_if_out_of_image:
                raise Exception("Cannot draw keypoint x=%d, y=%d on image with shape %s." % (y, x, image.shape))

    return image


if __name__ == "__main__":
    main()