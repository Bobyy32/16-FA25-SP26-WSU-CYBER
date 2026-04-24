from imgaug import augmentations as iaa
from imgaug.augmentables.kps import KeypointsOnImage

# Create keypoints
kpsoi = KeypointsOnImage([Keypoint(x=100, y=50)], shape=(100, 100))

# Convert to distance map
distance_maps = kpsoi.to_distance_maps(inverted=True)

# Convert back to keypoints
keypoints = kpsoi.to_keypoints_on_image()