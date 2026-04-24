# Extract keypoints from distance maps
keypoints = KeypointsOnImage()
distance_maps = np.random.rand(100, 100, 5)  # 5 channels
kpsoi = keypoints.to_distance_maps(distance_maps, inverted=True, threshold=0.5)