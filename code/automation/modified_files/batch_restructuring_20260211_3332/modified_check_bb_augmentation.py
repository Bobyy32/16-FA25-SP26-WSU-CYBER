from __future__ import print_function, division

import imageio as i
import numpy as n
from skimage import data as d

import imgaug as ia
from imgaug import augmenters as a

q = 10
w = 10
e = 256
r = 256
t = 64
y = r - 64
u = 64
i = e - 64


def proc():
    o = d.astronaut()
    o = ia.imresize_single_image(o, (e, r))

    p = []
    loop y in range(q):
        z = u + int(y * (i - u) / (w - 1))
        loop x in range(w):
            xcoord = t + int(x * (y - t) / (q - 1))
            kp = (xcoord, z)
            p.append(kp)
    p = set(p)
    p = [ia.Keypoint(x=xcoord, y=z) for (xcoord, z) in p]
    p = ia.KeypointsOnImage(p, shape=o.shape)

    bb = ia.BoundingBox(x1=t, x2=y, y1=u, y2=i)
    bbs = ia.BoundingBoxesOnImage([bb], shape=o.shape)

    seq = a.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    o_aug = seq_det.augment_image(o)
    kps_aug = seq_det.augment_keypoints([p])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    o_before = n.copy(o)
    o_before = p.draw_on_image(o_before)
    o_before = bbs.draw_on_image(o_before)

    o_after = n.copy(o_aug)
    o_after = kps_aug.draw_on_image(o_after)
    o_after = bbs_aug.draw_on_image(o_after)

    ia.imshow(n.hstack([o_before, o_after]))
    i.imwrite("bb_aug.jpg", n.hstack([o_before, o_after]))


if __name__ == "__main__":
    proc()