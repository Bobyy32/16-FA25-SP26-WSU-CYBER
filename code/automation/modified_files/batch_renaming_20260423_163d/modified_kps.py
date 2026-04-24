# Convert keypoints to distance maps
dist_maps = keypoints.to_distance_maps(
    threshold=None,
    inverted=True
)

# Extract keypoints from distance maps
kpsoi = distance_maps_to_keypoints_on_image(
    distance_maps=dist_maps,
    inverted=True,
    threshold=100
)