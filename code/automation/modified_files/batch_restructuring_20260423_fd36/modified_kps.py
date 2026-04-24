import numpy as np

# Convert distance maps to keypoints
kps = KeypointsOnImage.to_keypoints_on_image(distance_maps, inverted=True)

# Get a specific keypoint
keypoint = kps[0]  # x=..., y=...

# Iterate through keypoints
for kp in kps:
    print(kp.x, kp.y)

# Copy instance
kps_copy = kps.copy()