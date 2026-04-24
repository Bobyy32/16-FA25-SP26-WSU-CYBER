import imgaug

# Create heatmap from uint8 array
heatmap_uint8 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
hm = imgaug.augmentables.heatmaps.HeatmapsOnImage.from_uint8(
    arr_uint8=heatmap_uint8,
    shape=(100, 100),
    min_value=0.0,
    max_value=255.0
)

# Convert to float 0-1
arr_0to1 = hm.get_arr()  # Returns float array in [0.0, 1.0] range

# Rescale to [-1.0, 1.0]
rescaled = hm.change_normalization(source=(0.0, 1.0), target=(-1.0, 1.0))