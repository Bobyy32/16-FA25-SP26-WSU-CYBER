import numpy as np
import imgaug

# Create distance maps (example with 3 keypoints)
distance_maps = np.random.rand(224, 224, 3)

# Extract keypoints from distance maps
kpsoi = distance_maps_to_keypoints_on_image(distance_maps)

# Copy keypoints
copied = kpsoi.copy()

# Iterate over keypoints
for i, kp in enumerate(kpsoi):
    print(f"Keypoint {i}: x={kp.x}, y={kp.y}")