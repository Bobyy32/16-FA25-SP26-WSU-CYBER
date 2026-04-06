# Create heatmaps from uint8 data
heatmaps = HeatmapsOnImage.from_uint8(uint8_array, image_shape)

# Draw on image
overlay = heatmaps.draw_on_image(image, alpha=0.5)

# Get actual values in desired range
arr = heatmaps.get_arr()  # Returns array in [min_value, max_value] range