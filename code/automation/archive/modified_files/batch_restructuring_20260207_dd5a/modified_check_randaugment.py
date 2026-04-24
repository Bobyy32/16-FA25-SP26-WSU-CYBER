from __future__ import print_function, division, absolute_import

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


def q():
    s = ia.data.quokka(0.25)

    o = [1, 2]
    for N in o:
        print("N=%d" % (N,))

        d = []
        m = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        p = 0
        while p < len(m):
            u = m[p]
            d.extend(
                iaa.RandAugment(n=N, m=u, random_state=1)(images=[s] * 10)
            )
            p += 1
        ia.imshow(ia.draw_grid(d, cols=10))

    g = [0, 1, 2, 4, 8, 10]
    for M in g:
        print("M=%d" % (M,))
        t = iaa.RandAugment(m=M, random_state=1)

        v = []
        w = 0
        while w < 6:
            v.extend(t(images=[s] * 16))
            w += 1

        ia.imshow(ia.draw_grid(v, cols=16, rows=6))


if __name__ == "__main__":
    q()