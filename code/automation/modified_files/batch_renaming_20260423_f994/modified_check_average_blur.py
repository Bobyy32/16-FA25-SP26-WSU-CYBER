from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

unique_time_per_step_1 = 5000
unique_nb_augs_per_image_1 = 10


def unique_main_1():
    unique_image_1 = data.astronaut()
    unique_image_2_1 = ia.imresize_single_image(unique_image_1, (64, 64))
    print("unique_image_2_1 shape:", unique_image_2_1.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (unique_time_per_step_1,))

    unique_k_1 = [
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

    cv2.namedWindow("unique_aug_1_1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("unique_aug_1_1", 64*unique_nb_augs_per_image_1, 64)

    for unique_ki_1 in unique_k_1:
        unique_aug_1_1 = iaa.AverageBlur(k=unique_ki_1)
        unique_img_aug_1_1 = [unique_aug_1_1.augment_image(unique_image_2_1) for _ in range(unique_nb_augs_per_image_1)]
        unique_img_aug_1_1 = np.hstack(unique_img_aug_1_1)
        print("unique_img_aug_1_1 dtype", unique_img_aug_1_1.dtype, "unique_npa_avg_1", unique_npa_avg_1(unique_img_aug_1_1, axis=tuple(range(0, unique_img_aug_1_1.ndim-1))))

        unique_title_1_1 = "unique_ki_1=%s" % (str(unique_ki_1),)
        unique_img_aug_1_1 = ia.draw_text(unique_img_aug_1_1, x=5, y=5, text=unique_title_1_1)

        cv2.imshow("unique_aug_1_1", unique_img_aug_1_1[..., ::-1])  # here with rgb2bgr
        cv2.waitKey(unique_time_per_step_1)


if __name__ == "__main__":
    unique_main_1()