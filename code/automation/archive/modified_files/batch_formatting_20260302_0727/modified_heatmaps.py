# Create heatmaps from uint8 array
heatmaps = HeatmapsOnImage.from_uint8(uint8_array, shape=image_shape)

# Convert to different value range
normalized = HeatmapsOnImage.change_normalization(heatmaps_arr, (0.0, 1.0), (-1.0, 1.0))

# Overlay on image
result_image = heatmaps.draw_on_image(image)