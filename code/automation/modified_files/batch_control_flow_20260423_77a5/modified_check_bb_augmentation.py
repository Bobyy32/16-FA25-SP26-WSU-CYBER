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

_coord_mapping = lambda x_idx, y_idx: (int(x_idx * (BB_X2 - BB_X1) / (NB_ROWS - 1)), int(y_idx * (BB_Y2 - BB_Y1) / (NB_COLS - 1)))

def compute_all_coords():
    coords = {
        (y, x): _coord_mapping(x, y) for y in range(NB_ROWS) for x in range(NB_COLS)
    }
    return set(coords.values())

def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    def extract_keypoints_from_coords(coords):
        return [ia.Keypoint(x=x, y=y) for x, y in coords]

    kps = set(compute_all_coords())
    kps = extract_keypoints_from_coords(kps)
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

    if len(image_before.shape) == len(image_after.shape):
        result = np.hstack([image_before, image_after])
    else:
        result = image_before

    imageio.imwrite("bb_aug.jpg", result)


if __name__ == "__main__":
    main()