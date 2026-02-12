This code defines the `HeatmapsOnImage` class, which is used to represent and manipulate heatmaps that are placed on top of images. Here's a breakdown of its key features:

## Main Purpose
The class represents heatmaps (probability or confidence maps) that are overlaid on images, commonly used in computer vision tasks like object detection, segmentation, or attention visualization.

## Key Features

### Constructor & Initialization
- Takes a float32 array with values in [0.0, 1.0] range
- Stores the image shape it's associated with
- Maintains value range information (min_value, max_value) for conversion

### Core Methods

**`get_arr()`** - Returns the underlying heatmap array
- Converts from [0.0, 1.0] to [min_value, max_value] range
- Handles both 2D and 3D arrays (H,W and H,W,C formats)

**`draw_on_image()`** - Visualizes heatmaps on images
- Can overlay heatmaps with different colors (red, blue, green)
- Supports alpha blending for transparency
- Can draw multiple heatmaps simultaneously

**`draw()`** - Creates visualization without image context
- Returns RGB heatmaps with color mapping
- Supports different color schemes (red, blue, green, gray, jet)

**`blend()`** - Combines heatmaps with images
- Uses alpha blending for overlay effects
- Supports different blending modes

### Conversion Methods
- `to_uint8()` - Converts to 0-255 integer range
- `from_uint8()` - Creates from uint8 arrays
- `from_0to1()` - Creates from [0.0, 1.0] float arrays
- `change_normalization()` - Transforms value ranges

### Image Processing Operations
- `resize()` - Resizes heatmaps to match image dimensions
- `invert()` - Flips heatmap values (1.0 - value)
- `normalize()` - Ensures values are in [0.0, 1.0] range
- `clip()` - Clips values to [0.0, 1.0] range

### Utility Methods
- `copy()` and `deepcopy()` - For object duplication
- `__repr__()` - String representation for debugging

## Use Cases
This class is particularly useful for:
- Visualizing attention maps in neural networks
- Object detection heatmap visualization
- Segmentation mask overlay
- Debugging computer vision models
- Creating heatmaps for visualization in research papers or presentations

The implementation handles both single and multi-channel heatmaps, making it flexible for various computer vision applications.