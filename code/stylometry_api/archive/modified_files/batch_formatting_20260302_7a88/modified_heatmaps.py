# Create heatmaps from uint8 array
heatmaps = HeatmapsOnImage.from_uint8(arr_uint8, shape=image_shape)

# Draw on image
overlay = heatmaps.draw_on_image(image, alpha=0.5, colormap='jet')

# Convert between value ranges
normalized = HeatmapsOnImage.change_normalization(arr, (0,1), (-1,1))