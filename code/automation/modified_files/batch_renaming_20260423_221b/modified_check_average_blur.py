from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

TIME = 5000
NB = 10


def process():
    data = data.astronaut()
    data = ia.imresize_single_image(data, (64, 64))
    print("image shape:", data.shape)
    print("Press any key or wait %d ms to proceed to the next item." % (TIME,))

    item = [
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

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", 64*NB, 64)

    for i in item:
        process = iaa.AverageBlur(k=i)
        result = [process.augment_image(data) for _ in range(NB)]
        result = np.hstack(result)
        print("dtype", result.dtype, "averages", np.average(result, axis=tuple(range(0, result.ndim-1))))

        title = "k=%s" % (str(i),)
        result = ia.draw_text(result, x=5, y=5, text=title)

        cv2.imshow("aug", result[..., ::-1])  # here with rgb2bgr
        cv2.waitKey(TIME)


if __name__ == "__main__":
    process()