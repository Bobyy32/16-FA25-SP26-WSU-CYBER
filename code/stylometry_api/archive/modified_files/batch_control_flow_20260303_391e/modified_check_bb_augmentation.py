from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

N_ROWS = 10
N_COLS = 10
H_HEIGHT = 256
W_WIDTH = 256
B_B_X1 = 64
B_B_X2 = W_WIDTH - 64
B_B_Y1 = 64
B_B_Y2 = H_HEIGHT - 64


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (H_HEIGHT, W_WIDTH))

    kps = []
    for y in range(N_ROWS):
        ycoord = B_B_Y1 + int(y * (B_B_Y2 - B_B_Y1) / (N_COLS - 1))
        for x in range(N_COLS):
            xcoord = B_B_X1 + int(x * (B_B_X2 - B_B_X1) / (N_ROWS - 1))
            kp = (xcoord, ycoord)
            kps.append(kp)
    kps = set(kps)
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    kps = ia.KeypointsOnImage(kps, shape=image.shape)

    bb = ia.BoundingBox(x1=B_B_X1, x2=B_B_X2, y1=B_B_Y1, y2=B_B_Y2)
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