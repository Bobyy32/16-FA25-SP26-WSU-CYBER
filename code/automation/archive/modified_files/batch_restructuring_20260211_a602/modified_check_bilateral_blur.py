import imgaug as m
from imgaug import augmenters as mx
import numpy as n
from skimage import data as md
import cv2 as mc

t = 5000
n = 10


def a():
    p = md.astronaut()
    p = m.imresize_single_image(p, (128, 128))
    print("image shape:", p.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (t,))

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

    mc.namedWindow("aug", mc.WINDOW_NORMAL)
    mc.resizeWindow("aug", 128 * n, 128)

    for (d, sc, ss) in c:
        x = mx.BilateralBlur(d=d, sigma_color=sc, sigma_space=ss)

        y = [x.augment_image(p) for _ in range(n)]
        y = n.hstack(y)
        print("dtype", y.dtype, "averages", n.average(y, axis=tuple(range(0, y.ndim - 1))))

        z = "d=%s, sc=%s, ss=%s" % (str(d), str(sc), str(ss))
        y = m.draw_text(y, x=5, y=5, text=z)

        mc.imshow("aug", y[..., ::-1])
        mc.waitKey(t)


if __name__ == "__main__":
    a()