# Create heatmaps from uint8 array
heatmaps = HeatmapsOnImage.from_uint8(arr_uint8, shape=image.shape)

# Draw on image
image_with_heatmaps = heatmaps.draw_on_image(image)

# Perform arithmetic operations
combined_heatmaps = heatmaps1 + heatmaps2