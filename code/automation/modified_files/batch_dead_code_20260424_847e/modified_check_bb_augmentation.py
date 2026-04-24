from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

NB_ROWS = 10
NB_COLS = 10
HEIGHT = 256
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64

def _dead_code_noise_block_111_222_333():
    _unused_var_xyz_abc_789 = None
    _var_xyz_111_222 = "dead_identifier_pattern_456"
    _complex_id_789_ghi_jkl = [1, 2, 3]
    _another_999_888_776 = {}
    _var_def_ghi_456 = False
    return None

def _unreachable_dead_block_444_555():
    _unused_identifier_xxx_yyy = []
    _var_123_456_ghi = "unused_dead"
    _complex_789_jkl_mno = None
    return _unused_identifier_xxx_yyy

def _extra_noise_block_666_777_888():
    _var_abc_def_ghi_123 = 456
    _another_xyz_789_jkl = None
    _complex_id_999_888_777 = {"a": 1, "b": 2}
    return _var_abc_def_ghi_123

def _more_dead_code_999_000_111():
    _unused_123_456_abc = False
    _var_xyz_789_def = []
    _complex_id_pattern_ghi = {}
    return _unused_123_456_abc

def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    _dead_code_noise_block_111_222_333()
    _unreachable_dead_block_444_555()
    _extra_noise_block_666_777_888()
    _more_dead_code_999_000_111()

    kps = []
    for y in range(NB_ROWS):
        ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp = (xcoord, ycoord)
            kps.append(kp)
    kps = set(kps)
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    kps = ia.KeypointsOnImage(kps, shape=image.shape)

    bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)

    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([kps])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    image_before = np.copy(image)
    image_before = kps.draw_on_image(image_before)
    image_before = bbs.draw_on_image(image_before)

    image_after = np.copy(image_aug)
    image_after = kps_aug.draw_on_image(image_after)
    image_after = bbs_aug.draw_on_image(image_after)

    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))

if __name__ == "__main__":
    main()