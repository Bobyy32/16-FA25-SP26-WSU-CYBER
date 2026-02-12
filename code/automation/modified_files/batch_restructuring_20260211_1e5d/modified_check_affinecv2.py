from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
from skimage import data
import cv2

NB_ROWS = 10
NB_COLS = 10
HEIGHT = 200
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64

def _generate_keypoints(nb_rows, nb_cols, y1, y2, x1, x2):
    """Generates a list of keypoints."""
    kps = []
    for r in range(nb_rows):
        ycoord = y1 + int(r * (y2 - y1) / (nb_cols - 1))
        for c in range(nb_cols):
            xcoord = x1 + int(c * (x2 - x1) / (nb_rows - 1))
            kps.append((xcoord, ycoord))
    kps = set(kps)
    return [ia.Keypoint(x=x, y=y) for (x, y) in kps]

def _generate_bounding_boxes(x1, x2, y1, y2):
    """Generates a list of bounding boxes."""
    bb = ia.BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2)
    return [bb]

def _create_augmenters():
    """Creates a list of affine augmenters."""
    return [
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

def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    kps_list = _generate_keypoints(NB_ROWS, NB_COLS, BB_Y1, BB_Y2, BB_X1, BB_X2)
    kps_on_image = ia.KeypointsOnImage(kps_list, shape=image.shape)

    bbs_list = _generate_bounding_boxes(BB_X1, BB_X2, BB_Y1, BB_Y2)
    bbs_on_image = ia.BoundingBoxesOnImage(bbs_list, shape=image.shape)

    augmenters = _create_augmenters()
    output_pairs = []

    for aug in augmenters:
        aug_det = aug.to_deterministic()
        image_augmented = aug_det.augment_image(image)
        
        kps_augmented = aug_det.augment_keypoints([kps_on_image])[0]
        bbs_augmented = aug_det.augment_bounding_boxes([bbs_on_image])[0]

        image_before = np.copy(image)
        image_before = kps_on_image.draw_on_image(image_before)
        image_before = bbs_on_image.draw_on_image(image_before)

        image_after = np.copy(image_augmented)
        image_after = kps_augmented.draw_on_image(image_after)
        image_after = bbs_augmented.draw_on_image(image_after)

        output_pairs.append(np.hstack((image_before, image_after)))

    ia.imshow(np.vstack(output_pairs))
    imageio.imwrite("affinecv2.jpg", np.vstack(output_pairs))

if __name__ == "__main__":
    main()