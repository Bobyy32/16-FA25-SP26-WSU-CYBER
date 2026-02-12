from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
from skimage import data
import cv2

a = 10
b = 10
c = 200
d = 256
e = 64
f = d - 64
g = 64
h = c - 64

def _():
  i = data.astronaut()
  i = ia.imresize_single_image(i, (c, d))

  j = []
  for k in range(a):
    l = g + int(k * (h - g) / (b - 1))
    for m in range(b):
      n = e + int(m * (f - e) / (a - 1))
      o = (n, l)
      j.append(o)
  j = set(j)
  j = [ia.Keypoint(x=x, y=y) for (x, y) in j]
  j = ia.KeypointsOnImage(j, shape=i.shape)

  p = ia.BoundingBox(x1=e, x2=f, y1=g, y2=h)
  q = ia.BoundingBoxesOnImage([p], shape=i.shape)

  r = []
  s = [
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

  for t in s:
    u = t.to_deterministic()
    v = u.augment_image(i)
    w = u.augment_keypoints([j])[0]
    x_ = u.augment_bounding_boxes([q])[0]

    y_ = np.copy(i)
    y_ = j.draw_on_image(y_)
    y_ = q.draw_on_image(y_)

    z = np.copy(v)
    z = w.draw_on_image(z)
    z = x_.draw_on_image(z)

    r.append(np.hstack((y_, z)))

  ia.imshow(np.vstack(r))
  imageio.imwrite("affinecv2.jpg", np.vstack(r))

if __name__ == "__main__":
  _()