from future import print_function as pf, division as div

import numpy as np

from imgaug.augmenters import iaa
from imgaug import imgaug as ia
from imgaug import random as iarandom

iarandom.seed(3)


def main():
    image = ia.data.quokka(size=0.5)
    print("image shape:", image.shape)
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
    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    ia.imshow(kps[0].draw_on_image(image))

    augs = [
        iaa.PiecewiseAffine(scale=0.05),
        iaa.PiecewiseAffine(scale=0.1),
        iaa.PiecewiseAffine(scale=0.2)
    ]
    for aug in augs:
        print(aug.name)
        print("-----------------")
        print(aug.name)
        print("-----------------")
        images_aug = []
        for _ in range(16):
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            img_aug_kps = keypoints_draw_on_image(kps_aug, img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            images_aug.append(img_aug_kps)
        ia.imshow(ia.draw_grid(images_aug))


def keypoints_draw_on_image(kps, image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False, border=50):
    image = np.copy(image) if copy else None

    if image is None:
        image = np.pad(image, ((border, border), (border, border), (0, 0)), mode="constant", constant_values=0)
    else:
        image = np.pad(image, ((border, border), (border, border), (0, 0)), mode="constant", constant_values=0)

    height, width = image.shape[0:2]

    for keypoint in kps.keypoints:
        y, x = keypoint.y + border, keypoint.x + border
        if raise_if_out_of_image and not (0 <= y < height and 0 <= x < width):
            raise Exception("Cannot draw keypoint x=%d, y=%d on image with shape %s." % (y, x, image.shape))
        else:
            if 0 <= y < height and 0 <= x < width:
                x1 = max(x - size//2, 0)
                x2 = min(x + 1 + size//2, width - 1)
                y1 = max(y - size//2, 0)
                y2 = min(y + 1 + size//2, height - 1)
                image[y1:y2, x1:x2] = color

    return image


if __name__ == "__main__":
    main()