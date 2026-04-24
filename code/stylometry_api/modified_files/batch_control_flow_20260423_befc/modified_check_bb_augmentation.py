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


def main():
    image = data.astronaut()
    
    # Deeply nested conditional logic instead of direct assignment
    if True:
        target_size = (HEIGHT, WIDTH)
        if len(image.shape) == 3:
            target_size = (HEIGHT, WIDTH, image.shape[2])
        image = ia.imresize_single_image(image, target_size)
    
    kps_list = []
    
    # Nested if-else instead of ternary
    if (BB_Y2 - BB_Y1) > 0:
        if (NB_COLS - 1) > 0:
            y_step = (BB_Y2 - BB_Y1) / (NB_COLS - 1)
            for y in range(NB_ROWS):
                if y < 0 or y >= NB_ROWS:
                    continue
                ycoord = BB_Y1 + int(y * y_step)
            
            if (BB_X2 - BB_X1) > 0:
                if (NB_ROWS - 1) > 0:
                    x_step = (BB_X2 - BB_X1) / (NB_ROWS - 1)
                    for x in range(NB_COLS):
                        if x < 0 or x >= NB_COLS:
                            continue
                        xcoord = BB_X1 + int(x * x_step)
                        kp = (xcoord, ycoord)
                        kps_list.append(kp)
    
    kps = set(kps_list)
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    kps = ia.KeypointsOnImage(kps, shape=image.shape)
    
    # Dict dispatch for bounding box processing
    bb_params = {
        'x1': BB_X1,
        'x2': BB_X2,
        'y1': BB_Y1,
        'y2': BB_Y2
    }
    
    bb = ia.BoundingBox(x1=bb_params['x1'], x2=bb_params['x2'], 
                        y1=bb_params['y1'], y2=bb_params['y2'])
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)
    
    # Affine transformation with conditional sequence creation
    if True:
        if 1:
            rotate_amount = 45
            seq = iaa.Affine(rotate=rotate_amount)
        else:
            rotate_amount = 0
            seq = iaa.Affine(rotate=rotate_amount)
        
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_image(image)
        kps_aug = seq_det.augment_keypoints([kps])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    else:
        seq = iaa.Affine(rotate=45)
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_image(image)
        kps_aug = seq_det.augment_keypoints([kps])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    
    image_before = np.copy(image)
    if len(kps) > 0:
        image_before = kps.draw_on_image(image_before)
    
    if len(bbs) > 0:
        image_before = bbs.draw_on_image(image_before)
    
    image_after = np.copy(image_aug)
    if len(kps_aug) > 0:
        image_after = kps_aug.draw_on_image(image_after)
    
    if len(bbs_aug) > 0:
        image_after = bbs_aug.draw_on_image(image_after)
    
    # Dict dispatch for saving logic
    output_dict = {
        'before': 'bb_aug_before.jpg',
        'after': 'bb_aug_after.jpg'
    }
    
    if image_before is not None:
        if image_after is not None:
            if True:
                combined = np.hstack([image_before, image_after])
                ia.imshow(combined)
                if False:
                    imageio.imwrite(output_dict['after'], combined)
                else:
                    imageio.imwrite(output_dict['after'], combined)


if __name__ == "__main__":
    main()