# Create keypoints on an image
kpsoi = KeypointsOnImage(
    keypoints=[Keypoint(x=100, y=50), Keypoint(x=150, y=50)],
    shape=(200, 200)
)

# Access specific keypoints
print(kpsoi[0])  # Output: Keypoint(x=100, y=50)

# Iterate over keypoints
for kp in kpsoi:
    print(kp.x, kp.y)

# Convert from distance map
from imgaug.augmentables.kps import KeypointsOnImage
dm = ... # distance map
kpsoi = KeypointsOnImage.to_distance_maps(dm, if_not_found_coords=(99, 99))