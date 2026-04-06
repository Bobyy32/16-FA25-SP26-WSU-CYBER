# Create from uint8 data
heatmaps = HeatmapsOnImage.from_uint8(uint8_array, shape=image_shape)

# Convert to different value range
normalized_heatmaps = HeatmapsOnImage.change_normalization(
    heatmaps.get_arr(), (0.0, 1.0), (-1.0, 1.0)
)

# Draw on image
result_image = heatmaps.draw_on_image(image, alpha=0.5)