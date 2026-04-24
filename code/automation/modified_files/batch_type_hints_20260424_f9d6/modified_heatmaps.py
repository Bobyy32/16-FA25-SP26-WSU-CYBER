# Create a heatmap in [0.0, 1.0]
hm = HeatmapsOnImage.from_0to1(
    np.random.rand(100, 100),
    shape=(100, 100),
    min_value=0.0, max_value=1.0
)

# Convert to uint8
uint8_hm = hm.to_uint8()

# Change the value range
new_hm = HeatmapsOnImage.change_normalization(
    hm.get_arr(),
    (0.0, 1.0),
    (-1.0, 1.0)
)