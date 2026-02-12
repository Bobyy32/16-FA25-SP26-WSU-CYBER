# Create heatmaps from uint8 data
heatmaps = HeatmapsOnImage.from_uint8(uint8_array, image_shape)

# Convert to different value range
normalized = HeatmapsOnImage.change_normalization(arr, (0,255), (-1,1))

# Draw on image
overlay = heatmaps.draw_on_image(image, alpha=0.5)