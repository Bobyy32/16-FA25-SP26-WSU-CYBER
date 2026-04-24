from imgaug.augmentables.heatmaps import HeatmapsOnImage
import numpy as np

# Create heatmap with value range [-1.0, 1.0]
heatmap = np.array([[0.0], [1.0]], dtype=np.float32)
hm = HeatmapsOnImage.from_0to1(heatmap, shape=(32, 32), min_value=-1.0, max_value=1.0)

# Convert to uint8
uint8_map = hm.to_uint8()

# Rescale to [0.0, 1.0]
rescaled = HeatmapsOnImage.change_normalization(
    arr=hm.get_arr(),
    source=(hm.min_value, hm.max_value),
    target=(0.0, 1.0)
)