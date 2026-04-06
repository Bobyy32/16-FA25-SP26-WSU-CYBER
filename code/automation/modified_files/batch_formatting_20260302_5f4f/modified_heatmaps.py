This code defines the `HeatmapsOnImage` class, which is used to represent and manipulate heatmap data associated with images in the imgaug library. Here's a breakdown of its key features:

## Main Purpose
The class represents heatmaps that are placed on top of images, commonly used for object detection, segmentation, or attention visualization.

## Key Features

### Core Properties
- **`arr_0to1`**: Internal float array with values in [0.0, 1.0] range
- **`shape`**: Shape of the image the heatmaps are associated with
- **`min_value`/`max_value`**: Actual value range of the heatmaps (default [0.0, 1.0])

### Creation Methods
- `from_uint8()`: Creates heatmaps from uint8 arrays (0-255 range)
- `from_0to1()`: Creates heatmaps from float arrays (0.0-1.0 range)
- `change_normalization()`: Converts between different value ranges

### Visualization & Conversion
- `draw_on_image()`: Overlays heatmaps on images
- `to_uint8()`: Converts to uint8 format
- `get_arr()`: Returns the underlying array in the specified value range

### Image Operations
- `resize()`: Resizes heatmaps to match image dimensions
- `pad()`: Pads heatmaps with zeros
- `crop()`: Crops heatmaps to a region
- `normalize()`: Normalizes values to [0.0, 1.0]

### Mathematical Operations
- `__add__()`, `__sub__()`, `__mul__()`, `__truediv__()`: Element-wise arithmetic
- `__neg__()`: Negation
- `__abs__()`: Absolute value
- `__pow__()`: Power operations

### Utility Methods
- `copy()`/`deepcopy()`: Shallow and deep copying
- `draw()`: Draws heatmaps as colored overlays
- `get_arr()` and `get_arr_normalized()`: Access to different value representations

The class is designed to be flexible for various computer vision tasks while maintaining proper value range handling and efficient operations.