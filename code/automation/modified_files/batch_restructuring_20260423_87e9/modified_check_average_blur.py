from __future__ import print_function, division

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data

TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10

def process_kernels(kernels, image, augmentations_per):
    augmented = []
    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", 64 * augmentations_per, 64)

    for k in kernels:
        aug = iaa.AverageBlur(k=k)
        img_aug = [aug.augment_image(image) for _ in range(augmentations_per)]
        img_aug = np.hstack(img_aug)
        print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))

        title = "k=%s" % str(k)
        img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)

        cv2.imshow("aug", img_aug[..., ::-1])
        cv2.waitKey(TIME_PER_STEP)

        augmented.append(img_aug)

    return augmented

def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (64, 64))
    print("image shape:", image.shape)
    print("Press any key or wait %d ms to proceed to the next image." % TIME_PER_STEP)

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

    processed = process_kernels(k, image, NB_AUGS_PER_IMAGE)