from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

from typing import List, Tuple, Optional, Union, Iterable

TIME_PER_STEP: int = 5000
NB_AUGS_PER_IMAGE: int = 10


def main(image: np.ndarray) -> None:
    image: np.ndarray = ia.imresize_single_image(image, Tuple[int, int])
    print("image shape:", image.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (TIME_PER_STEP,))

    k: Tuple[Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]] = [
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
    cv2.resizeWindow("aug", int, int, int, int)

    for ki:
        aug = iaa.AverageBlur(k: Union[int, Tuple[int, int]])
        img_aug: List[np.ndarray] = [aug.augment_image(image) for _ in range(NB_AUGS_PER_IMAGE)]
        img_aug: np.ndarray = np.hstack(img_aug)
        print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))

        title: str = "k=%s" % (str(ki),)
        img_aug: np.ndarray = ia.draw_text(img_aug, x: int, y: int, text: str)

        cv2.imshow("aug", img_aug[..., ::-1])
        cv2.waitKey(TIME_PER_STEP)


if __name__ == "__main__":
    main()