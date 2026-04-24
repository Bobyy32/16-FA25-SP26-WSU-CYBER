from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10


def augment_image(img, k_val, augment_count):
    aug = iaa.AverageBlur(k=k_val)
    img_aug = [aug.augment_image(img) for _ in range(augment_count)]
    img_aug = np.hstack(img_aug)
    return img_aug


def display_augment_results(img_aug, title):
    print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))
    
    img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)
    
    cv2.imshow("aug", img_aug[..., ::-1])  # here with rgb2bgr
    cv2.waitKey(TIME_PER_STEP)


def prepare_image(image):
    image = ia.imresize_single_image(image, (64, 64))
    return image


def initialize_display():
    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", 64*NB_AUGS_PER_IMAGE, 64)


def main():
    global TIME_PER_STEP, NB_AUGS_PER_IMAGE
    
    image = data.astronaut()
    image = prepare_image(image)
    print("image shape:", image.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (TIME_PER_STEP,))
    
    k = [
        1,
        2,
        4,
        8,
        16,
        (8, 8),
        (1, 8),
        ((1, 1), (8, 8)),
        ((1, 16), (1, 16)),
        ((1, 16), 1)
    ]
    
    initialize_display()

    for ki in k:
        img_aug = augment_image(image, ki, NB_AUGS_PER_IMAGE)
        title = "k=%s" % (str(ki),)
        display_augment_results(img_aug, title)


if __name__ == "__main__":
    main()