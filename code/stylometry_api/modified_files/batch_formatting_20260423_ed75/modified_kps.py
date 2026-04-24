distance_maps = np.random.rand(100, 100, 5).astype('float32')
kpsoi = keypoints.to_distance_maps(distance_maps, inverted=True)