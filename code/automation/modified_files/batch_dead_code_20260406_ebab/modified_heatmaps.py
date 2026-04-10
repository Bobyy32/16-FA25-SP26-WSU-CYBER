# Create from normalized float array
h = HeatmapsOnImage.from_0to1(heatmap_arr, shape)

# Convert to uint8 for visualization
h_uint8 = h.to_uint8()

# Change value range
h_new = HeatmapsOnImage.change_normalization(h.get_arr(), (0, 1), (-1, 1))

# Create deep copy for processing
h_copy = h.deepcopy()