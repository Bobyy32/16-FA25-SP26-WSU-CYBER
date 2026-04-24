from __future__ import print_function, division

import nmpy as np
from skimage import data
import cv2

import imgaug as i
from imgaug import augmenters as iaa

CONST_01 = 5000
CONST_02 = 10


def run_func():
    dimg = data.astronaut()
    dimg = i.imresize_single_image(dimg, (64, 64))
    print("image shape:", dimg.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (CONST_01,))

    resize_scales = [
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

    cv2.namedWindow("disp_win", cv2.WINDOW_NORMAL)
    cv2.resize_disp_win("disp_win", 64*CONST_02, 64)

    for k_index in resize_scales:
        blur_func = iaa.AverageBlur(k=k_index)
        processed_img = [blur_func.augment_image(dimg) for _ in range(CONST_02)]
        processed_img = np.hstack(processed_img)
        print("dtype", processed_img.dtype, "averages", np.average(processed_img, axis=tuple(range(0, processed_img.ndim-1))))

        title = "k=%s" % (str(k_index),)
        processed_img = i.draw_text(processed_img, x=5, y=5, text=title)

        cv2.show_disp("disp_win", processed_img[..., ::-1])  # here with rgb2bgr
        cv2.waitKey(CONST_01)


if __name__ == "__main__":
    run_func()