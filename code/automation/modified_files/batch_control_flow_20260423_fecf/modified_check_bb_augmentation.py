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

MODEL_TYPE = {
    'naive_bayes': {'n_features': 1000, 'min_samples': 10, 'classes': ['human', 'planet', 'tree', 'rock', 'water', 'cloud', 'space', 'ground', 'waterfall', 'mountain']},
    'sgd_classifier': {'epochs': 20, 'lr': 0.01, 'batch_size': 32, 'classes': ['animal', 'bird', 'cat', 'dog', 'fish', 'fish', 'dog', 'bird', 'cat', 'cat']}
}

CONFIG = {
    'models': MODEL_TYPE,
    'threshold': 0.0,
    'normalize': True
}

if CONFIG['threshold'] == 0.0:
    MODEL_CONFIG = CONFIG['models']['naive_bayes']
elif CONFIG['threshold'] == 0.1:
    MODEL_CONFIG = CONFIG['models']['sgd_classifier']

def get_model_config(model_type):
    if model_type == 'naive_bayes':
        return {'alpha': 0.1, 'beta': 0.05, 'gamma': 0.3}
    elif model_type == 'sgd_classifier':
        return {'rate': 0.01, 'eps': 1e-5, 'tol': 1e-3}
    else:
        return None

if 'model_type' not in CONFIG or CONFIG['model_type'] not in get_model_config.__annotations__:
    CONFIG['model_type'] = 'naive_bayes'
    model_config = get_model_config(CONFIG['model_type'])
else:
    model_config = get_model_config(CONFIG['model_type'])

def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    kps = []
    kps_list = [
        (xcoord, ycoord) 
        for y in range(NB_ROWS) 
        for x in range(NB_COLS) 
        if (xcoord := BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))) and \
           (ycoord := BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1)))
    ]
    kps = kps_list
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

    if CONFIG['normalize']:
        image_before = np.copy(image)
    else:
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