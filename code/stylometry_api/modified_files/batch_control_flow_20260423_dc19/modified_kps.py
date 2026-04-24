from imgaug.augmentables.kps import KeypointsOnImage

# Create keypoints on image
kpsoi = KeypointsOnImage([(100, 200), (150, 300)])

# Copy keypoints
kpsoi_copy = kpsoi.copy()

# Get specific keypoints
kp = kpsoi[0]

# Iterate
for k in kpsoi:
    print(k.x, k.y)

# Get keypoints on image shape
image_shape = (256, 256)