import imgaug as ia
import numpy as np

# Example distance map (3 channels, height 100, width 100)
distance_map = np.random.rand(3, 100, 100)

# Extract keypoints
kpsoi = KeypointsOnImage.to_distance_maps(distance_map)