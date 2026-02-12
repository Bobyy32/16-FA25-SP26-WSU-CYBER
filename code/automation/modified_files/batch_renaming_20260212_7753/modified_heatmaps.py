This code defines a `HeatmapsOnImage` class for handling heatmap data in image processing tasks. Here's a breakdown of its key features:

## Main Purpose
The class represents heatmaps that are "on" images - meaning they're overlayed on top of images for visualization or analysis purposes.

## Key Features

### Core Properties
- **`arr_0to1`**: Internal float array with values in [0.0, 1.0] range
- **`shape`**: Shape of the image the heatmap is placed on
- **`min_value`/`max_value`**: Actual value range the heatmap represents

### Creation Methods
- `from_uint8()`: Creates heatmap from uint8 array (0-255 range)
- `from_0to1()`: Creates heatmap from float array (0.0-1.0 range)
- `change_normalization()`: Converts between different value ranges

### Visualization & Processing
- `draw_on_image()`: Overlay heatmap on image with optional transparency
- `draw()`: Draw heatmap as colored overlay
- `get_arr()`: Returns the actual heatmap array in the specified value range
- `to_uint8()`: Convert to uint8 format (0-255)

### Image Operations
- `resize()`: Scale the heatmap to different dimensions
- `invert()`: Flip the heatmap values (0→1, 1→0)
- `normalize()`: Normalize values to [0.0, 1.0] range

### Utility Methods
- `copy()`/`deepcopy()`: Create copies of the heatmap
- `__repr__()`/`__str__()`: String representations for debugging

## Use Cases
This class is particularly useful for:
- Semantic segmentation visualization
- Attention map overlays
- Heatmap generation for computer vision tasks
- Debugging and visualization of model outputs

The design allows for flexible value range handling while maintaining efficient internal representation in the [0.0, 1.0] range for computation.