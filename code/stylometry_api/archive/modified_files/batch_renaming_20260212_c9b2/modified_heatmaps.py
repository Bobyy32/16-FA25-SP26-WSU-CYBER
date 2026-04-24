This code defines a `HeatmapsOnImage` class for handling heatmaps that are placed on images. Here's a breakdown of its key features:

## Main Purpose
The class represents heatmaps (probability/activation maps) that are overlayed on images, commonly used in computer vision tasks like object detection, segmentation, or attention visualization.

## Key Features

### Core Attributes
- `arr_0to1`: Stores heatmap data as float32 values in range [0.0, 1.0]
- `shape`: Original image shape (H, W, C) where heatmap is placed
- `min_value/max_value`: Actual value range the heatmap represents (e.g., [0.0, 1.0] or [-1.0, 1.0])

### Main Methods

**Creation Methods:**
- `from_uint8()`: Creates heatmap from uint8 array (0-255 range)
- `from_0to1()`: Creates heatmap from float array in [0.0, 1.0] range
- `change_normalization()`: Converts between different value ranges

**Visualization/Display:**
- `draw_on_image()`: Overlays heatmap on image with optional transparency
- `draw()`: Creates visualization with color mapping
- `get_arr()`: Returns actual heatmap values in specified range

**Manipulation:**
- `invert()`: Flips heatmap values (1.0 → 0.0, etc.)
- `blend()`: Combines heatmaps with blending
- `resize()`: Changes heatmap dimensions
- `normalize()`: Ensures values are in [0.0, 1.0] range

**Utility:**
- `to_uint8()`: Converts to 0-255 uint8 format
- `copy()/deepcopy()`: Creates copies of heatmap objects

### Key Design Choices
1. **Separation of concerns**: Stores normalized [0.0, 1.0] data internally but can represent any value range
2. **Flexible input**: Handles both 2D (H,W) and 3D (H,W,C) arrays
3. **Image overlay**: Can draw heatmaps directly on images with transparency control
4. **Value range conversion**: Supports conversion between different value ranges (e.g., [0,1] to [-1,1])

This class is particularly useful for visualization tasks in deep learning, where you might want to see attention maps, feature activations, or prediction confidence overlays on images.