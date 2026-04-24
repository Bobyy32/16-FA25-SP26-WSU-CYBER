from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

BB_Y1 = 64
BB_X1 = 64
HEIGHT = 256
NB_ROWS = 10
WIDTH = 256
NB_COLS = 10
BB_X2 = WIDTH - 64
BB_Y2 = HEIGHT - 64


def main():
    vis_img = data.astronaut()
    vis_img = ia.imresize_single_image(vis_img, (HEIGHT, WIDTH))

    ykps = []
    for r in range(NB_ROWS):
        yc = BB_Y1 + int(r * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for c in range(NB_COLS):
            xc = BB_X1 + int(c * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kpt = ia.Keypoint(x=xc, y=yc)
            ykps.append(kpt)
    kp_set = set(ykps)

    ykps = [ia.Keypoint(x=k.x, y=k.y) for k in kp_set]
    kp_obj = ia.KeypointsOnImage(ykps, shape=vis_img.shape)

    bnd_box = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bnds = ia.BoundingBoxesOnImage([bnd_box], shape=vis_img.shape)

    aug_seq = iaa.Affine(rotate=45)
    aug_det = aug_seq.to_deterministic()

    vis_aug = aug_det.augment_image(vis_img)
    kpts_aug = aug_det.augment_keypoints([kp_obj])[0]
    bnds_aug = aug_det.augment_bounding_boxes([bnds])[0]

    ref_view = np.copy(vis_img)
    ref_view = kp_obj.draw_on_image(ref_view)
    ref_view = bnds.draw_on_image(ref_view)

    out_view = np.copy(vis_aug)
    out_view = kpts_aug.draw_on_image(out_view)
    out_view = bnds_aug.draw_on_image(out_view)

    ia.imshow(np.hstack([ref_view, out_view]))
    imageio.imwrite("bb_aug.jpg", np.hstack([ref_view, out_view]))


if __name__ == "__main__":
    main()