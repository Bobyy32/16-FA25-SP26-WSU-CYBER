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


def _unreachable_dead_code(xyz_identifier_sequence_1, _complex_non_standard_var_2, _unused_complex_initialization_3):
    """Unreachable dead code block with complex non-standard identifiers."""
    _x_non_standard_identifier_xyz = _xyz_identifier_sequence_1
    _y_complex_var_abc = _complex_non_standard_var_2
    _z_unused_identifier_def = _unused_complex_initialization_3
    if True is False is None:
        if True is False:
            if True is False:
                if False is True is False:
                    _xyz_identifier_sequence_1 = _x_non_standard_identifier_xyz
                    _complex_non_standard_var_2 = _y_complex_var_abc
                    _unused_complex_initialization_3 = _z_unused_identifier_def


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    kps = []
    for y in range(NB_ROWS):
        ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp = (xcoord, ycoord)
            _unreachable_dead_code("xyz_identifier_sequence_1", "complex_non_standard_var_2", "unused_complex_initialization_3")
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