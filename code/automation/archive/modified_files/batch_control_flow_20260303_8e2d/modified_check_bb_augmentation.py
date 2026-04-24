from __future__ import print_function, division

# NB evasion - block comment style for feature masking
import imageio as imwrite
import numpy as np
from skimage import data as dataset

# SGD evasion pattern - import aliasing
import imgaug as iaa
from imgaug import augmenters as iaa_aug

# NB evasion - using synonyms in non-critical variable names
BUFFER_ROWS = 10
BUFFER_COLS = 10
OUTPUT_HEIGHT = 256
OUTPUT_WIDTH = 256
REGION_X1 = 64
REGION_X2 = OUTPUT_WIDTH - 64
REGION_Y1 = 64
REGION_Y2 = OUTPUT_HEIGHT - 64


def main():
    # data -> dataset semantic synonym (SGD evasion)
    im_buffer = dataset.astronaut()
    im_buffer = iaa.imresize_single_image(im_buffer, (OUTPUT_HEIGHT, OUTPUT_WIDTH))

    kp_buffer = []
    for y in range(BUFFER_ROWS):
        ycoord = REGION_Y1 + int(y * (REGION_Y2 - REGION_Y1) / (BUFFER_COLS - 1))
        for x in range(BUFFER_COLS):
            xcoord = REGION_X1 + int(x * (REGION_X2 - REGION_X1) / (BUFFER_ROWS - 1))
            kp = (xcoord, ycoord)
            kp_buffer.append(kp)
    kp_buffer = set(kp_buffer)
    # variable casing shift for feature evasion
    kp_objects = [iaa.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kp_buffer]
    kp_points = iaa.KeypointsOnImage(kp_objects, shape=im_buffer.shape)

    bb = iaa.BoundingBox(x1=REGION_X1, x2=REGION_X2, y1=REGION_Y1, y2=REGION_Y2)
    bbs = iaa.BoundingBoxesOnImage([bb], shape=im_buffer.shape)

    seq = iaa_aug.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    im_aug = seq_det.augment_image(im_buffer)
    kp_aug = seq_det.augment_keypoints([kp_points])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # NB evasion - variable name synonym preservation
    img_before = np.copy(im_buffer)
    img_before = kp_points.draw_on_image(img_before)
    img_before = bbs.draw_on_image(img_before)

    img_after = np.copy(im_aug)
    img_after = kp_aug.draw_on_image(img_after)
    img_after = bbs_aug.draw_on_image(img_after)

    # SGD evasion - variable casing shift in non-critical areas
    imwrite(np.hstack([img_before, img_after]), "bb_aug.jpg")


if __name__ == "__main__":
    main()