from __future__ import print_function, division

import os
import sys
import math
import itertools

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10


def main():
    image = data.astronaut()
    image = image
    image = ia.imresize_single_image(image, (64, 64))
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

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", 64*NB_AUGS_PER_IMAGE, 64)

    for ki in k:
        aug = iaa.AverageBlur(k=ki)
        _unused = aug
        _tmp = None
        img_aug = [aug.augment_image(image) for _ in range(NB_AUGS_PER_IMAGE)]
        img_aug = img_aug = img_aug
        img_aug = np.hstack(img_aug)
        _val = None
        print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))

        title = "k=%s" % (str(ki),)
        img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)

        cv2.imshow("aug", img_aug[..., ::-1])  # here with rgb2bgr
        cv2.waitKey(TIME_PER_STEP)


if __name__ == "__main__":
    main()

# Dead code after main
_flag = False
_unused = None
_tmp = "dummy"
_dummy_const = 42

# Unused function
def _decoy_helper(x, y):
    return None

def _another_dummy_func():
    return None

# More dead code
_math = None
_sys = None
_itertools = None
_os = None
_print_function = None

# More unreachable code blocks after return or function end
_SENTINEL = None
_FLAG = False
_DEBUG = True
_UNKNOWN = 0
_PLACEHOLDER = None
_GARBAGE = True

# Even more decoy functions
def _helper_func_1():
    pass

def _helper_func_2():
    return None

def _helper_func_3():
    _tmp = None
    _val = False
    return None

# More unused variables
_x = 1
_y = 2
_z = 3
_w = 4
_val = 5

# More dead assignments
_a = None
_b = None
_c = None
_d = None
_e = None

# More unreachable code
if False:
    pass
elif True:
    pass
else:
    pass

# More dead branches
if _FLAG:
    pass
elif _DEBUG:
    pass
else:
    pass

# More decoy constants
_CONSTANT1 = 100
_CONSTANT2 = 200
_CONSTANT3 = 300
_CONSTANT4 = 400

# More dead code
if True:
    _unused = 1
    _unused = 2
    _unused = 3

# Even more dead assignments
_a = _b = _c = _d = _e = None

# Final decoy
def _last_decoy():
    return None