# Create from uint8 array
heatmaps = HeatmapsOnImage.from_uint8(arr_uint8, shape=(height, width))

# Convert between value ranges
normalized = HeatmapsOnImage.change_normalization(arr, (0,255), (-1,1))

# Visualize
overlay = heatmaps.draw_on_image(image)