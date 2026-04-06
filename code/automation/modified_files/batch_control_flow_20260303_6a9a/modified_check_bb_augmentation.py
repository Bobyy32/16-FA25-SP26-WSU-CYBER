from __future__ import print_function, division

import imageio as II
import numpy as np
from skimage import data

import imgaug as IA
from imgaug import augmenters as iaa

Nb_ROWS = 10
Nb_COLS = 10
Height = 256
Width = 256
BB_X_1 = 64
BB_X_2 = Width - 64
BB_Y_1 = 64
BB_Y_2 = Height - 64


def Main():
    img = data.astronaut()
    img = IA.imresize_single_image(img, (Height, Width))

    Kp = []
    for Y in range(Nb_ROWS):
        y_coord = BB_Y_1 + int(Y * (BB_Y_2 - BB_Y_1) / (Nb_COLS - 1))
        for X in range(Nb_COLS):
            x_coord = BB_X_1 + int(X * (BB_X_2 - BB_X_1) / (Nb_ROWS - 1))
            keyPoint = (x_coord, y_coord)
            Kp.append(keyPoint)
    Kp = set(Kp)
    Kp = [IA.Keypoint(x=x_coord, y=y_coord) for (x_coord, y_coord) in Kp]
    Kp = IA.KeypointsOnImage(Kp, shape=img.shape)

    bb = IA.BoundingBox(x1=BB_X_1, x2=BB_X_2, y1=BB_Y_1, y2=BB_Y_2)
    bbs = IA.BoundingBoxesOnImage([bb], shape=img.shape)

    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_image(img)
    kps_aug = seq_det.augment_keypoints([Kp])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    Image_before = np.copy(img)
    Image_before = Kp.draw_on_image(Image_before)
    Image_before = bbs.draw_on_image(Image_before)

    Image_after = np.copy(image_aug)
    Image_after = kps_aug.draw_on_image(Image_after)
    Image_after = bbs_aug.draw_on_image(Image_after)

    II.imshow(np.hstack([Image_before, Image_after]))
    II.imwrite("bb_aug.jpg", np.hstack([Image_before, Image_after]))


if __name__ == "__main__":
    Main()