# Create heatmaps from uint8 array
heatmaps = HeatmapsOnImage.from_uint8(heatmap_array, shape=image_shape)

# Draw on image
overlay = heatmaps.draw_on_image(image, alpha=0.5, colormap='jet')

# Convert between value ranges
normalized_heatmaps = HeatmapsOnImage.change_normalization(
    heatmaps_array, (0.0, 1.0), (-1.0, 1.0)
)