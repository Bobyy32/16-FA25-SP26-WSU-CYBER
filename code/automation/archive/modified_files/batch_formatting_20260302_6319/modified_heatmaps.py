# Create heatmaps from uint8 array
heatmaps = HeatmapsOnImage.from_uint8(arr_uint8, image_shape)

# Overlay on image
result = heatmaps.draw_on_image(image, alpha=0.5)

# Resize heatmaps
resized = heatmaps.resize((new_height, new_width))