This code defines a `HeatmapsOnImage` class for handling heatmaps in image processing tasks. Here's a breakdown of its key features:

## Main Purpose
The class represents heatmaps that are "on" an image, storing both the heatmap data and the image dimensions they correspond to.

## Key Methods

### Constructor & Initialization
- `__init__(self, arr_0to1, shape, min_value=0.0, max_value=1.0)`
- `from_uint8()` and `from_0to1()` - Factory methods to create heatmaps from different data types
- `change_normalization()` - Converts between different value ranges

### Data Access
- `get_arr()` - Returns the underlying heatmap array
- `to_uint8()` - Converts to uint8 format [0,255]
- `__getitem__()` - Allows array-style indexing

### Visualization & Display
- `draw_on_image()` - Overlays heatmaps on images with alpha blending
- `draw()` - Creates visualization with color mapping
- `draw_as_overlay()` - Creates overlay visualization
- `draw_on_image_with_alpha()` - Draws with customizable alpha

### Image Operations
- `invert()` - Inverts heatmap values
- `blend()` - Combines heatmaps with alpha blending
- `normalize()` - Normalizes values to [0,1] range
- `resize()` - Resizes heatmaps to match image dimensions

### Mathematical Operations
- `__add__`, `__sub__`, `__mul__`, `__truediv__` - Element-wise arithmetic
- `__pow__` - Power operations
- `__neg__` - Negation
- `__abs__` - Absolute values

### Utility Methods
- `copy()` and `deepcopy()` - Object copying
- `get_shape()` - Returns heatmap dimensions
- `get_arr()` - Returns the actual heatmap data array

## Key Features
1. **Flexible Input**: Can handle different data formats (uint8, float [0,1], float [min,max])
2. **Image Integration**: Stores image dimensions and can overlay heatmaps on images
3. **Value Range Management**: Tracks and converts between different value ranges
4. **Visualization**: Multiple drawing options for heatmaps
5. **Mathematical Operations**: Supports element-wise arithmetic operations
6. **Memory Efficient**: Uses shallow/deep copy patterns

The class is designed for computer vision applications where heatmaps need to be visualized on images, combined with other heatmaps, or processed mathematically while maintaining proper value ranges and spatial relationships.