import imgaug

# Load or generate keypoints on image
img = imgaug.augmentations.imgaug.Image("example.jpg")
kpsoi = img.augmentations.to_keypoints_on_image(img.detect("features"))

# Convert to distance maps for detection
distance_maps = kpsoi.to_distance_maps()

# Filter keypoints by distance threshold
threshold = 10.0
kpsoi_filtered = kpsoi.to_keypoints_on_image(
    distance_maps,
    threshold=threshold,
    if_not_found_coords=(-1, -1)  # Placeholder
)

# Access and iterate
for i, kp in enumerate(kpsoi_filtered):
    print(f"Keypoint {i}: ({kp.x}, {kp.y})")

# Copy keypoints
kp_copy = kpsoi_filtered.copy()
kp_deep_copy = kpsoi_filtered.deepcopy()

# Apply transformations
kpsoi_filtered.invert_to_keypoints_on_image(kpsoi_filtered)