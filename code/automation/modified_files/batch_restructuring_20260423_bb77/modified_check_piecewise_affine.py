from __future__ import print_function, division

import numpy as np
from imgaug.random import seed
import imgaug as ia
from imgaug import augmenters as iaa
import imgaug.random as iarandom
import imgaug.augmenters as ia_aug

iarandom.seed(3)


def draw_keypoints_on_image(kps, img, color=(0, 255, 0), sz=3, cp=True, raise_out=False, bd=50):
    if cp:
        img = np.copy(img)
    
    img = np.pad(img, ((bd, bd), (bd, bd), (0, 0)), mode="constant", constant_values=0)
    
    h, w = img.shape[0:2]
    
    for kp in kps.keypoints:
        x, y = kp.x, kp.y
        y += bd
        x += bd
        
        if 0 <= y < h and 0 <= x < w:
            x1 = max(x - sz//2, 0)
            x2 = min(x + 1 + sz//2, w - 1)
            y1 = max(y - sz//2, 0)
            y2 = min(y + 1 + sz//2, h - 1)
            img[y1:y2, x1:x2] = color
        else:
            if raise_out:
                raise Exception(f"Cannot draw keypoint x={x}, y={y} on image with shape {img.shape}.")
    
    return img


def main():
    img_orig = ia.data.quokka(size=0.5)
    print(f"image shape: {img_orig.shape}")

    kps = ia.KeypointsOnImage(
        [
            ia.Keypoint(x=123, y=102),
            ia.Keypoint(x=182, y=98),
            ia.Keypoint(x=155, y=134),
            ia.Keypoint(x=-20, y=20)
        ],
        shape=img_orig.shape
    )

    print("-----------------")
    print("Random aug per image")
    print("-----------------")

    aug_list = [
        iaa.PiecewiseAffine(scale=0.05),
        iaa.PiecewiseAffine(scale=0.1),
        iaa.PiecewiseAffine(scale=0.2)
    ]

    img_kps = kps.draw_on_image(img_orig)
    ia.imshow(img_kps)

    for aug in aug_list:
        augs_list = []
        for _ in range(16):
            aug_det = aug.to_deterministic()
            img_a = aug_det.augment_image(img_orig)
            kps_a = aug_det.augment_keypoints(kps)[0]
            img_akps = draw_keypoints_on_image(kps_a, img_a, border=50)
            img_p = np.pad(img_akps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            augs_list.append(img_p)
        
        print(aug.name)
        ia.imshow(ia.draw_grid(augs_list))


if __name__ == "__main__":
    main()