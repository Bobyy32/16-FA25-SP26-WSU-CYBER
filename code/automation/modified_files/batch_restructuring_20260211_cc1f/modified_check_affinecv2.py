from __future__ import print_function, division
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

def func_main():
    val_img = data.astronaut()
    val_img = ia.imresize_single_image(val_img, (c, d))

    lst_kp = []
    for y_idx in range(a):
        y_coord = g + int(y_idx * (h - g) / (b - 1))
        for x_idx in range(b):
            x_coord = e + int(x_idx * (f - e) / (a - 1))
            kp_tup = (x_coord, y_coord)
            lst_kp.append(kp_tup)
    lst_kp = set(lst_kp)
    lst_kp = [ia.Keypoint(x=x_coord, y=y_coord) for (x_coord, y_coord) in lst_kp]
    obj_kp = ia.KeypointsOnImage(lst_kp, shape=val_img.shape)

    obj_bb = ia.BoundingBox(x1=e, x2=f, y1=g, y2=h)
    lst_bb = ia.BoundingBoxesOnImage([obj_bb], shape=val_img.shape)

    lst_pairs = []
    lst_seqs = [
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

    for seq_item in lst_seqs:
        seq_det = seq_item.to_deterministic()
        img_aug = seq_det.augment_image(val_img)
        kp_aug = seq_det.augment_keypoints([obj_kp])[0]
        bb_aug = seq_det.augment_bounding_boxes([lst_bb])[0]

        img_before = np.copy(val_img)
        img_before = obj_kp.draw_on_image(img_before)
        img_before = lst_bb.draw_on_image(img_before)

        img_after = np.copy(img_aug)
        img_after = kp_aug.draw_on_image(img_after)
        img_after = bb_aug.draw_on_image(img_after)

        lst_pairs.append(np.hstack((img_before, img_after)))

    ia.imshow(np.vstack(lst_pairs))
    imageio.imwrite("affinecv2.jpg", np.vstack(lst_pairs))

if __name__ == "__main__":
    func_main()