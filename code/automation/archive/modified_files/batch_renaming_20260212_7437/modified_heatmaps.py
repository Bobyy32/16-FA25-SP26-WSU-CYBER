This code defines a `HeatmapsOnImage` class for handling heatmaps in image processing applications. Here's a breakdown of its key features:

## Main Purpose
The class represents heatmaps that are "on" or associated with images, storing both the heatmap data and metadata about the image they correspond to.

## Key Methods

### Constructor & Initialization
- `__init__`: Sets up the heatmap with data array and image shape
- `from_uint8`/`from_0to1`: Factory methods to create heatmaps from different data formats

### Data Handling
- `get_arr()`: Returns the actual heatmap data in the specified value range
- `to_uint8()`: Converts heatmap to 8-bit integer format [0,255]
- `change_normalization()`: Transforms value ranges (e.g., [0,1] to [-1,1])

### Visualization & Display
- `draw_on_image()`: Overlays heatmaps on images with optional alpha blending
- `draw()`: Creates visualization of heatmaps as RGB images
- `draw_on_image_heatmap()`: Draws heatmaps directly on images

### Image Processing Operations
- `invert()`: Flips heatmap values (0→1, 1→0)
- `blend()`: Combines heatmaps with alpha blending
- `normalize()`: Ensures values are in [0,1] range
- `resize()`: Changes heatmap dimensions

### Mathematical Operations
- `__add__`, `__sub__`, `__mul__`, `__truediv__`: Element-wise arithmetic operations
- `__neg__`: Negation operation
- `__abs__`: Absolute value operation

### Utility Functions
- `copy()`/`deepcopy()`: Creates copies of heatmap objects
- `__repr__`/`__str__`: String representations for debugging

## Key Features
1. **Flexible Input**: Accepts various data formats (uint8, float32)
2. **Value Range Management**: Tracks and converts between different value ranges
3. **Image Integration**: Maintains association with original images
4. **Visualization**: Easy overlay and display capabilities
5. **Mathematical Operations**: Supports arithmetic operations on heatmaps
6. **Memory Management**: Provides both shallow and deep copy options

This class is particularly useful for computer vision tasks involving heatmap generation, visualization, and manipulation, such as in segmentation, attention maps, or saliency detection applications.