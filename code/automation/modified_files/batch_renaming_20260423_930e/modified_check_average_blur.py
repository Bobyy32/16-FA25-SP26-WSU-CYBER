from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

dt_p_s = 5000
n_b_a_p_i = 10


def main():
    d_img_01_ = data.astronaut()
    d_img_01_ = ia.imresize_single_image(d_img_01_, (64, 64))
    print("image shape:", d_img_01_.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (dt_p_s,))

    l_ks_ = [
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

    v_c_n_w_ = "aug"
    v_c_w_n_ = 64 * n_b_a_p_i
    cv2.resizeWindow(v_c_w_n_, 64)

    for x_k_i_ in l_ks_:
        bl_a_ = iaa.AverageBlur(k=x_k_i_)
        d_im_a_ = [bl_a_.augment_image(d_img_01_) for _ in range(n_b_a_p_i)]
        d_im_a_ = np.hstack(d_im_a_)
        print("dtype", d_im_a_.dtype, "averages", np.average(d_im_a_, axis=tuple(range(0, d_im_a_.ndim-1))))

        t_t = "k=%s" % (str(x_k_i_),)
        d_im_a_ = ia.draw_text(d_im_a_, x=5, y=5, text=t_t)

        cv2.imshow(v_c_n_w_, d_im_a_[..., ::-1])  # here with rgb2bgr
        cv2.waitKey(dt_p_s)


if __name__ == "__main__":
    main()