# Create from uint8 array
heatmap = HeatmapsOnImage.from_uint8(arr_uint8, shape=image.shape)

# Convert to different value range
heatmap_normalized = HeatmapsOnImage.change_normalization(
    heatmap.get_arr(), (0.0, 1.0), (-1.0, 1.0)
)

# Draw on image
overlay = heatmap.draw_on_image(image, alpha=0.5)