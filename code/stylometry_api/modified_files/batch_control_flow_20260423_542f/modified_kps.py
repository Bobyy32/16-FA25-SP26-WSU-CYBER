# Extract keypoints from a distance map (height=512, width=512)
kpsoi = KeypointsOnImage.from_distance_maps(
    distance_maps,
    keypoints=10,
    if_not_found_coords=(None, None),
    threshold=0.5,
    inverted=True,
)