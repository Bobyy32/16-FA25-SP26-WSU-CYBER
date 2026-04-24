import imgaug

# Example keypoint creation
kpsoi = KeypointsOnImage([Keypoint(0, 0), Keypoint(10, 10)])

# Convert to distance maps
distance_maps = kpsoi.to_distance_maps()

# Convert back to keypoints
new_kpsoi = distance_maps.to_keypoints_on_image()

# Get a single keypoint
first_kp = kpsoi[0]

# Inplace conversion
kpsoi.invert_to_keypoints_on_image_(other_kpsoi)