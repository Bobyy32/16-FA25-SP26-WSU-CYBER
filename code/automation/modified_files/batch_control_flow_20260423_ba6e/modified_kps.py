import numpy as np
import imgaug.augmentables.kps

# Create distance map (3 channels = 3 keypoints)
distance_maps = np.array([
    [[100, 50], [50, 100]],
    [[20, 80], [90, 30]],
    [[15, 70], [65, 25]]
])

# Convert to keypoints
keypoints = KeypointsOnImage.to_distance_maps(
    distance_maps,
    if_not_found_coords=(100, 100),
    threshold=10,
    nb_channels=3
)

print(keypoints)  # Output: KeypointsOnImage(Keypoints... shape=(2, 2, 3))