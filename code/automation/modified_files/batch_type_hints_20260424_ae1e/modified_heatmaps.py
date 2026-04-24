from imgaug.augmentables import HeatmapsOnImage

# Create from normalized array (0.0–1.0)
hm = HeatmapsOnImage.from_0to1(arr, shape=(256, 256))

# Convert to uint8
uint8_arr = hm.to_uint8()

# Rescale values from [0, 1] to [-1, 1]
scaled_arr = HeatmapsOnImage.change_normalization(
    hm.get_arr(), (0.0, 1.0), (-1.0, 1.0))