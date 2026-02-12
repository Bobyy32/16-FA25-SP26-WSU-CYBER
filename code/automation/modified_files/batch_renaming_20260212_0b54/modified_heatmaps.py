# Create heatmaps from uint8 data
heatmaps = HeatmapsOnImage.from_uint8(uint8_array, shape=(height, width))

# Draw on image
overlay = heatmaps.draw_on_image(image_array, alpha=0.5)

# Convert between value ranges
normalized = HeatmapsOnImage.change_normalization(arr, (0,255), (-1,1))