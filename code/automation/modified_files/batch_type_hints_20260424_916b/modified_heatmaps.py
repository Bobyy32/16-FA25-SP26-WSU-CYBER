from imgaug import HeatmapsOnImage

# Convert uint8 to normalized float heatmap
heatmaps = HeatmapsOnImage.from_uint8(arr_uint8, shape=(img.shape))

# Adjust value range from [0, 1] to [-1, 1]
arr_new = HeatmapsOnImage.change_normalization(
    heatmaps.get_arr(),
    source=(0.0, 1.0),
    target=(-1.0, 1.0)
)