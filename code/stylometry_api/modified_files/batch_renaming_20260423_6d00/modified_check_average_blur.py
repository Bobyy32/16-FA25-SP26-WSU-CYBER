from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

time_per_step_001 = 5000
nb_augs_per_image_001 = 10


def main_executor_001():
    image_data_001 = data.astronaut()
    image_data_001 = ia.imresize_single_image(image_data_001, (64, 64))
    print("image shape:", image_data_001.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (time_per_step_001,))

    k_list_001 = [
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

    cv2.namedWindow("aug_001", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug_001", 64*nb_augs_per_image_001, 64)

    for ki_idx, ki_val_001 in enumerate(k_list_001):
        aug_001 = iaa.AverageBlur(k=ki_val_001)
        img_aug_001 = [aug_001.augment_image(image_data_001) for _ in range(nb_augs_per_image_001)]
        img_aug_001 = np.hstack(img_aug_001)
        print("dtype", img_aug_001.dtype, "averages", np.average(img_aug_001, axis=tuple(range(0, img_aug_001.ndim-1))))

        title_str_001 = "k=%s" % (str(ki_val_001),)
        img_aug_001 = ia.draw_text(img_aug_001, x=5, y=5, text=title_str_001)

        cv2.imshow("aug_001", img_aug_001[..., ::-1])
        cv2.waitKey(time_per_step_001)


if __name__ == "__main__":
    main_executor_001()