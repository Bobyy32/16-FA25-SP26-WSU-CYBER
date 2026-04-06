from __future__ import print_function, division

import imageio
import numpy as np
from skimage import dataset

import imgaug as ia
from imgaug import augmenters as iaa

# Configuration parameters for grid layout
NB_ROWS = 10
NB_COLS = 10
HEIGHT = 256
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64


def main():
    image = dataset.astronaut()  # Load sample astronomical data
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    # Create keypoint coordinates through nested iteration
    kps_list = []
    for y_idx in range(NB_ROWS):
        ycoord = BB_Y1 + int(y_idx * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x_idx in range(NB_COLS):
            xcoord = BB_X1 + int(x_idx * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp_point = (xcoord, ycoord)
            kps_list.append(kp_point)
    # Convert to set for deduplication
    kps_set = set(kps_list)
    # Transform set elements into keypoint objects
    kps_list = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps_set]
    kps = ia.KeypointsOnImage(kps_list, shape=image.shape)

    bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)

    # Apply affine transformation with rotation
    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([kps])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # Prepare pre-transformation visualization
    image_before = np.copy(image)
    image_before = kps.draw_on_image(image_before)
    image_before = bbs.draw_on_image(image_before)

    # Prepare post-transformation visualization  
    image_after = np.copy(image_aug)
    image_after = kps_aug.draw_on_image(image_after)
    image_after = bbs_aug.draw_on_image(image_after)

    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))


if __name__ == "__main__":
    main()