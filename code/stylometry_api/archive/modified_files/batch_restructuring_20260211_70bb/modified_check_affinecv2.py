from __future__ import print_function
from __future__ import division

import imgaug as  ia
from  imgaug  import augmenters  as  iaa
import imageio
import numpy  as  np
from   skimage  import  data
import cv2

a = 10
b = 10
c = 200
d = 256
e = 64
f =  d -  e
g = 64
h =  c -  g

def m():
    i =  data.astronaut()
    i =  ia.imresize_single_image(i, (c, d))

    j = []
    y = 0
    while y < a:
        ycoord = g + int(y * (h - g) / (b - 1))
        x = 0
        while x < b:
            xcoord = e + int(x * (f - e) / (a - 1))
            k = (xcoord, ycoord)
            j.append(k)
            x += 1
        y += 1
    j =  set(j)
    j = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in j]
    j =  ia.KeypointsOnImage(j, shape=i.shape)

    l = ia.BoundingBox(x1=e, x2=f, y1=g, y2=h)
    m = ia.BoundingBoxesOnImage([l], shape=i.shape)

    n = []
    o = [
        iaa.AffineCv2(rotate=45),
        iaa.AffineCv2(translate_px=20),
        iaa.AffineCv2(translate_percent=0.1),
        iaa.AffineCv2(scale=1.2),
        iaa.AffineCv2(scale=0.8),
        iaa.AffineCv2(shear=45),
        iaa.AffineCv2(rotate=45, cval=256),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_CONSTANT),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_REPLICATE),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_REFLECT),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_REFLECT_101),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_WRAP),
        iaa.AffineCv2(translate_px=20, mode="constant"),
        iaa.AffineCv2(translate_px=20, mode="replicate"),
        iaa.AffineCv2(translate_px=20, mode="reflect"),
        iaa.AffineCv2(translate_px=20, mode="reflect_101"),
        iaa.AffineCv2(translate_px=20, mode="wrap"),
        iaa.AffineCv2(scale=0.5, order=cv2.INTER_NEAREST),
        iaa.AffineCv2(scale=0.5, order=cv2.INTER_LINEAR),
        iaa.AffineCv2(scale=0.5, order=cv2.INTER_CUBIC),
        iaa.AffineCv2(scale=0.5, order=cv2.INTER_LANCZOS4),
        iaa.AffineCv2(scale=0.5, order="nearest"),
        iaa.AffineCv2(scale=0.5, order="linear"),
        iaa.AffineCv2(scale=0.5, order="cubic"),
        iaa.AffineCv2(scale=0.5, order="lanczos4"),
        iaa.AffineCv2(rotate=45, translate_px=20, scale=1.2),
        iaa.AffineCv2(rotate=45, translate_px=20, scale=0.8),
        iaa.AffineCv2(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL),
        iaa.AffineCv2(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL),
        iaa.AffineCv2(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL),
        iaa.AffineCv2(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL)
    ]

    for p in o:
        q = p.to_deterministic()
        r = q.augment_image(i)
        s = q.augment_keypoints([j])[0]
        t = q.augment_bounding_boxes([m])[0]

        u =  np.copy(i)
        u =  j.draw_on_image(u)
        u =  m.draw_on_image(u)

        v =  np.copy(r)
        v =  s.draw_on_image(v)
        v =  t.draw_on_image(v)

        n.append(np.hstack((u, v)))

    ia.imshow(np.vstack(n))
    imageio.imwrite("affinecv2.jpg", np.vstack(n))

if __name__ == "__main__":
    m()