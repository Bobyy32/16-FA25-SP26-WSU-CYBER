from __future__ import print_function, division
import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

image = data.astronaut()

sequence = ia.Sequence(ia.augmenters.Affine(), size=500, order=5)
image_aug = imageio.imread(image)

sequence_aug = ia.augmenters.Sequence(
    [ia.augmenters.Affine()], size=500, order=5
)

image_aug_det = image_aug.copy()
kps = set()
for kp in image_aug_det:
    kps.add(kp)

# Ensure set of keypoints
kps = set(kps)

# Ensure list of keypoints
kps = [ia.Keypoint]

# Get bounding box
bb = ia.BoundingBoxesOnImage([(10, 10, 100, 100)])
bbs = ia.BoundingBoxesOnImage([(10, 10, 100, 100)])

sequence_det = ia.Sequence(ia.augmenters.Affine(), size=500, order=5)

sequence_det_aug = ia.Sequence(
    [ia.augmenters.Affine()], size=500, order=5
)

image_before = imageio.imread(image)
image_after = imageio.imread(image)

# Apply augmentation to image
image_aug = sequence_aug.augment_image(image_before)
bbs_aug = image_aug.bbox

# Apply augmentation to keypoints
kps_aug = sequence_det.augment_sequence(kps)
bbs_aug = sequence_det_aug.augment_image(image_after)
bbs_aug = bbs_aug

imageio.imwrite(image_aug, image_before)