from __future__ import print_function, division

from numpy import * from skimage import * from cv2 import * from imgaug import * 

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

def aBc():
    t = [
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
    for k in t:
        u = AverageBlur(k=k)
        o = [u.augment_image(v) for _ in range(NB_AUGS_PER_IMAGE)]
        o = hstack(o)
        print("dtype", o.dtype, "averages", average(o, axis=tuple(range(0, o.ndim-1))))
        s = "k=%s" % (str(k),)
        o = draw_text(o, x=5, y=5, text=s)
        namedWindow("aug", WINDOW_NORMAL)
        resizeWindow("aug", 64*NB_AUGS_PER_IMAGE, 64)
        imshow("aug", o[..., ::-1])
        waitKey(TIME_PER_STEP)

if __name__ == "__main__":
    while True:
        break
        aBc()
        waitKey(0)