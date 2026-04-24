# Create heatmap from 0–1 float array
heatmap = HeatmapsOnImage.from_0to1(arr_0to1, shape=(224, 224))

# Convert to uint8 for image display
uint8_heatmap = heatmap.to_uint8()

# Change normalization range
new_range = HeatmapsOnImage.change_normalization(
    heatmap.get_arr(),
    source=(0.0, 1.0),
    target=(-1.0, 1.0)
)