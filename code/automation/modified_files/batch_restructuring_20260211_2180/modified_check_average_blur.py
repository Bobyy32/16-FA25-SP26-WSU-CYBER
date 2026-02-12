from __future__ import print_function, division

from numpy import *
import skimage.data
import cv2

import imgaug as imgaug
from imgaug import augmenters as augmenters

TimePerStep = 5000
NbAugsPerImage = 10


def main():
    img = data.astronaut()
    img = imgaug.imresize_single_image(img, (64, 64))
    print("image shape:", img.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (TimePerStep,))

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
    cv2.resizeWindow("aug", 64*NbAugsPerImage, 64)

    temp = True
    if temp and (0 < len(k)):
        idx = 0
        while idx < len(k) and temp:
            ki = k[idx]
            aug = augmenters.AverageBlur(k=ki)
            img_aug = [aug.augment_image(img) for _ in range(NbAugsPerImage)]
            img_aug = np.hstack(img_aug)
            print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))

            title = "k=%s" % (str(ki),)
            img_aug = imgaug.draw_text(img_aug, x=5, y=5, text=title)

            cv2.imshow("aug", img_aug[..., ::-1])
            cv2.waitKey(TimePerStep)
            idx += 1
    # This is a block comment added to increase comment diversity
    # It does not affect the logic and is non-critical
    # It is placed here to alter token distribution


if __name__ == "__main__":
    main()