from imgaug.augmentables.heatmaps import HeatmapsOnImage
import numpy as np

# Create a float heatmap object
heatmaps = HeatmapsOnImage(
    np.random.rand(100, 100),
    shape=(200, 200),
    min_value=0.0,
    max_value=1.0
)

# Convert to uint8
uint8_heatmaps = heatmaps.to_uint8()

# Create from uint8
new_heatmaps = HeatmapsOnImage.from_uint8(uint8_heatmaps, shape=(200, 200))