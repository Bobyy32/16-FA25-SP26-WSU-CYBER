import imgaug as ia
import imgaug.augmenters as iaa
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

def z():
    x = data.astronaut()
    x = ia.imresize_single_image(x, (c, d))

    y = []
    for q in range(a):
        w = g + int(q * (h - g) / (b - 1))
        for r in range(b):
            u = e + int(r * (f - e) / (a - 1))
            v = (u, w)
            y.append(v)
    y = set(y)
    y = [ia.Keypoint(x=k, y=l) for (k, l) in y]
    y = ia.KeypointsOnImage(y, shape=x.shape)

    s = ia.BoundingBox(x1=e, x2=f, y1=g, y2=h)
    t = ia.BoundingBoxesOnImage([s], shape=x.shape)

    p = []
    _ = [
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

    for m in _:
        n = m.to_deterministic()
        x_ = n.augment_image(x)
        y_ = n.augment_keypoints([y])[0]
        t_ = n.augment_bounding_boxes([t])[0]

        x_tmp = np.copy(x)
        x_tmp = y.draw_on_image(x_tmp)
        x_tmp = t.draw_on_image(x_tmp)

        x_final = np.copy(x_)
        x_final = y_.draw_on_image(x_final)
        x_final = t_.draw_on_image(x_final)

        p.append(np.hstack((x_tmp, x_final)))

    ia.imshow(np.vstack(p))
    imageio.imwrite("affinecv2.jpg", np.vstack(p))

if __name__ == "__main__":
    z()