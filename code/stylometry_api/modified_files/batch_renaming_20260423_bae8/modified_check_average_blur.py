from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2 as opencv_lib

import imgaug as img_lib
from imgaug import augmenters as img_aug_lib

STEP_TIME_MS = 5000
IMAGE_AUGMENT_COUNT = 10


def init_image():
    img = data.astronaut()
    img = img_lib.imresize_single_image(img, (64, 64))
    print("img shape:", img.shape)
    print("Press any key or wait %d ms to proceed to the next img." % (STEP_TIME_MS,))

    key_list = [
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

    opencv_lib.namedWindow("aug", opencv_lib.WINDOW_NORMAL)
    opencv_lib.resizeWindow("aug", 64*IMAGE_AUGMENT_COUNT, 64)

    for key in key_list:
        aug = img_aug_lib.AverageBlur(k=key)
        img_aug = [aug.augment_image(img) for _ in range(IMAGE_AUGMENT_COUNT)]
        img_aug = np.hstack(img_aug)
        print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))

        title = "k=%s" % (str(key),)
        img_aug = img_lib.draw_text(img_aug, x=5, y=5, text=title)

        opencv_lib.imshow("aug", img_aug[..., ::-1])
        opencv_lib.waitKey(STEP_TIME_MS)


if __name__ == "__main__":
    init_image()