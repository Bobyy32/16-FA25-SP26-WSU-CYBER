from imgaug.augmentables.heatmaps import HeatmapsOnImage
import numpy as np

# Create a 0-1 normalized heatmap
heatmaps = HeatmapsOnImage.from_0to1(
    np.random.rand(100, 100).astype(np.float32),
    shape=(100, 100),
    min_value=0.0,
    max_value=1.0
)

# Convert to uint8
uint8_heatmap = heatmaps.to_uint8()

# Rescale to [-1, 1]
rescaled = HeatmapsOnImage.change_normalization(
    heatmaps.get_arr(),
    source=(0.0, 1.0),
    target=(-1.0, 1.0)
)