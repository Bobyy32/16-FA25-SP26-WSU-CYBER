import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data
import cv2

T = 5000
N = 10


def main():
    i = data.astronaut()
    i = ia.imresize_single_image(i, (128, 128))
    print("image shape:", i.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (T,))

    c = [
        (1, 75, 75),
        (3, 75, 75),
        (5, 75, 75),
        (10, 75, 75),
        (10, 25, 25),
        (10, 250, 150),
        (15, 75, 75),
        (15, 150, 150),
        (15, 250, 150),
        (20, 75, 75),
        (40, 150, 150),
        ((1, 5), 75, 75),
        (5, (10, 250), 75),
        (5, 75, (10, 250)),
        (5, (10, 250), (10, 250)),
        (10, (10, 250), (10, 250)),
    ]

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", 128*N, 128)

    i = 0
    while i < len(c):
        (d, s_c, s_s) = c[i]
        b = iaa.BilateralBlur(d=d, sigma_color=s_c, sigma_space=s_s)
        t = [b.augment_image(i) for _ in range(N)]
        t = np.hstack(t)
        print("dtype", t.dtype, "averages", np.average(t, axis=tuple(range(0, t.ndim-1))))
        title = "d=%s, sc=%s, ss=%s" % (str(d), str(s_c), str(s_s))
        t = ia.draw_text(t, x=5, y=5, text=title)
        cv2.imshow("aug", t[..., ::-1]) # here with rgb2bgr
        cv2.waitKey(T)
        i += 1

if __name__ == "__main__":
    main()