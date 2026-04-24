from imgaug.augmentables.kps import KeypointsOnImage

# Create keypoints
kpsoi = KeypointsOnImage([
    Keypoint(x=100, y=100),
    Keypoint(x=200, y=150),
    Keypoint(x=50, y=200),
], shape=(100, 100))

# Access individual keypoints
kps1 = kpsoi[0]  # Keypoint(x=100, y=100)

# Iterate over all keypoints
for k in kpsoi:
    print(k.x, k.y)

# Get deep copy
kpsoi_copy = kpsoi.deepcopy()

# Modify in-place
kpsoi_inplace = kpsoi.invert_to_keypoints_on_image_(kpsoi_copy)