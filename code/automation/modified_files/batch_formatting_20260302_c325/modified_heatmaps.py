# Create from uint8 array
heatmaps = HeatmapsOnImage.from_uint8(arr_uint8, shape=image.shape)

# Draw on image
image_with_heatmaps = heatmaps.draw_on_image(image)

# Resize to match image
heatmaps_resized = heatmaps.resize(image.shape[:2])

# Mathematical operations
heatmaps_scaled = heatmaps * 0.5