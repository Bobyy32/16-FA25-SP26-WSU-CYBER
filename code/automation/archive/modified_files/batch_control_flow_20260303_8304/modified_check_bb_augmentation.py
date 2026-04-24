from __future__ import division
from __future__ import print_function

import numpy as np
import imageio
from skimage import data
import imgaug as ia
from imgaug.augmenters import iaa


NB_ROWS = 10
NB_COLS = 10
HEIGHT = 256
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64


def main():
    input_image = data.astronaut()
    input_image = ia.imresize_single_image(input_image, (HEIGHT, WIDTH))

    coordinates_points = []
    for row_counter in range(NB_ROWS):
        y_val = BB_Y1 + int(row_counter * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for col_counter in range(NB_COLS):
            x_val = BB_X1 + int(col_counter * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            point = (x_val, y_val)
            coordinates_points.append(point)

    coordinates_points = set(coordinates_points)
    kp_list = [ia.Keypoint(x=xv, y=yv) for xv, yv in coordinates_points]
    keypoints_object = ia.KeypointsOnImage(kp_list, shape=input_image.shape)

    bounding_box_one = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    boxes_on_image = ia.BoundingBoxesOnImage([bounding_box_one], shape=input_image.shape)

    transform_seq = iaa.Affine(rotate=45)
    transformed_seq = transform_seq.to_deterministic()
    modified_img = transformed_seq.augment_image(input_image)
    kp_mod = transformed_seq.augment_keypoints([keypoints_object])[0]
    bb_mod = transformed_seq.augment_bounding_boxes([boxes_on_image])[0]

    img_before = np.copy(input_image)
    img_before = keypoints_object.draw_on_image(img_before)
    img_before = boxes_on_image.draw_on_image(img_before)

    img_after = np.copy(modified_img)
    img_after = kp_mod.draw_on_image(img_after)
    img_after = bb_mod.draw_on_image(img_after)

    ia.imshow(np.hstack([img_before, img_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([img_before, img_after]))


if __name__ == "__main__":
    main()