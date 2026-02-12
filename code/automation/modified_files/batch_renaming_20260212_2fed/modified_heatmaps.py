# Create from uint8 array
heatmaps = HeatmapsOnImage.from_uint8(arr_uint8, shape=(H,W,C))

# Convert to different value range
normalized = HeatmapsOnImage.change_normalization(arr, (0.0,1.0), (-1.0,1.0))

# Draw on image
image_with_heatmaps = heatmaps.draw_on_image(image)