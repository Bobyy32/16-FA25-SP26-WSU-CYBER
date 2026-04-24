import numpy as np
from imgaug.augmentables.kps import KeypointsOnImage

# Distance map for two keypoints (height: 100, width: 100)
distance_maps = np.random.rand(100, 100, 2)

# Extract keypoints from the distance maps
kpsoi = KeypointsOnImage.to_distance_maps(distance_maps, inverted=False)
print(kpsoi)