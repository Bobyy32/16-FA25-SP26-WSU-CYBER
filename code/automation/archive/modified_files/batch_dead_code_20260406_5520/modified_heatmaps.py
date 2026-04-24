import numpy as np
from imgaug.augmentables import HeatmapsOnImage

# Create a heatmap object
hmap = HeatmapsOnImage(
    np.zeros((100, 100)),
    shape=(100, 100),
    min_value=0.0,
    max_value=1.0
)

# Convert to uint8
hmap_uint8 = hmap.to_uint8()

# Change normalization
hmap_new_range = HeatmapsOnImage.change_normalization(hmap, (0.0, 1.0), (-1.0, 1.0))

# Access the array
arr = hmap.get_arr()