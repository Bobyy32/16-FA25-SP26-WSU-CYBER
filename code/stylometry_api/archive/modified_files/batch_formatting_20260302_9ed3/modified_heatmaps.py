# Create from uint8 array
heatmaps = HeatmapsOnImage.from_uint8(arr_uint8, shape=(H,W))

# Convert to different value range
normalized = HeatmapsOnImage.change_normalization(arr, (0,1), (-1,1))

# Draw on image
overlay = heatmaps.draw_on_image(image)