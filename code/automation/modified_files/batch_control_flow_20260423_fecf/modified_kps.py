from imgaug import augmentations as iaa
import imgaug.augmentables.kps as kp

# Create keypoints container
kps = kp.KeypointsOnImage(keypoints, shape=(224, 224, 3))

# Use a distance map to extract keypoints
kps2 = kps.to_distance_maps(distance_maps, ...)

# Convert keypoints back to KeypointsOnImage
kps3 = kp.to_keypoints_on_image(kps2)

# Apply augmentations
aug = iaa.RandomFlip(horizontal=True)