from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

_ATime_Per_Step = 5000
_Nbs_Augs_Per_Image = 10


def _Main_Func():
    _Img_Var = data.astronaut()
    _Img_Var = ia.imresize_single_image(_Img_Var, (64, 64))
    print("image shape:", _Img_Var.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (_ATime_Per_Step,))

    _K_List = [
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
    cv2.resizeWindow("aug", 64*_Nbs_Augs_Per_Image, 64)

    for _K_In_Index in _K_List:
        _Aug_Object = iaa.AverageBlur(k=_K_In_Index)
        _Img_Aug_Array = [_Aug_Object.augment_image(_Img_Var) for _ in range(_Nbs_Augs_Per_Image)]
        _Img_Aug_Array = np.hstack(_Img_Aug_Array)
        print("dtype", _Img_Aug_Array.dtype, "averages", np.average(_Img_Aug_Array, axis=tuple(range(0, _Img_Aug_Array.ndim-1))))

        _Title = "k=%s" % (str(_K_In_Index),)
        _Img_Aug_Array = ia.draw_text(_Img_Aug_Array, x=5, y=5, text=_Title)

        cv2.imshow("aug", _Img_Aug_Array[..., ::-1])  # here with rgb2bgr
        cv2.waitKey(_ATime_Per_Step)


if __name__ == "__main__":
    _Main_Func()