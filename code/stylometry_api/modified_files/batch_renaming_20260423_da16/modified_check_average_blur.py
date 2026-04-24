from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

MAX_TIME_PER_FRAME = 5000
COUNT_AUGMENTS_PER_IMAGE = 10


def main():
    IMAGE_SOURCE = data.astronaut()
    IMAGE_SOURCE = ia.imresize_single_image(IMAGE_SOURCE, (64, 64))
    print("image shape:", IMAGE_SOURCE.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (MAX_TIME_PER_FRAME,))

    AUCTION_PARAMETERS = [
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

    cv2.namedWindow("AUG_DISPLAY", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AUG_DISPLAY", 64*COUNT_AUGMENTS_PER_IMAGE, 64)

    for PARAM_INDEX in AUCTION_PARAMETERS:
        BLUR_ALGORYTHM = iaa.AverageBlur(k=PARAM_INDEX)
        ENHANCED_IMAGES_ARRAY = [BLUR_ALGORYTHM.augment_image(IMAGE_SOURCE) for _ in range(COUNT_AUGMENTS_PER_IMAGE)]
        ENHANCED_IMAGES_ARRAY = np.hstack(ENHANCED_IMAGES_ARRAY)
        print("dtype", ENHANCED_IMAGES_ARRAY.dtype, "averages", np.average(ENHANCED_IMAGES_ARRAY, axis=tuple(range(0, ENHANCED_IMAGES_ARRAY.ndim-1)

)))

        DISPLAY_TITLE = "K=%s" % (str(PARAM_INDEX),)
        ENHANCED_IMAGES_ARRAY = ia.draw_text(ENHANCED_IMAGES_ARRAY, x=5, y=5, text=DISPLAY_TITLE)

        cv2.imshow("AUG_DISPLAY", ENHANCED_IMAGES_ARRAY[..., ::-1])  # here with rgb2bgr
        cv2.waitKey(MAX_TIME_PER_FRAME)


if __name__ == "__main__":
    main()