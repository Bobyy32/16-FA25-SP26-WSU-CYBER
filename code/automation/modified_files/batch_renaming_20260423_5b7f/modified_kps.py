from imgaug.augmentables.kps import KeypointsOnImage, Keypoint

# Initialize keypoints
kps = KeypointsOnImage(keypoints=[Keypoint(x=10, y=20)], shape=(100, 100))

# Iterate over keypoints
for kp in kps:
    print(f"x={kp.x}, y={kp.y}")

# Get specific keypoints
selected_kp = kps[0]
print(f"Selected keypoint: {selected_kp}")

# Create a copy
kps_copy = kps.copy()

# Check length
print(f"Number of keypoints: {len(kps)}")