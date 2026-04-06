This code defines a `HeatmapsOnImage` class for handling heatmap data in image processing tasks. Here's a breakdown of its key features:

## Main Purpose
The class represents heatmaps that are "on" images - meaning they're overlayed on top of images to show areas of interest, confidence, or other spatial data.

## Key Features

### Core Properties
- **`arr_0to1`**: Internal float array with values in [0.0, 1.0] range
- **`shape`**: Shape of the image the heatmaps are associated with
- **`min_value`/`max_value`**: Actual value range the heatmaps represent (e.g., [0.0, 1.0] or [-1.0, 1.0])

### Conversion Methods
- **`from_uint8()`**: Creates heatmaps from uint8 arrays (0-255 range)
- **`from_0to1()`**: Creates heatmaps from [0.0, 1.0] float arrays
- **`to_uint8()`**: Converts back to uint8 format
- **`change_normalization()`**: Changes value range (e.g., [0.0, 1.0] to [-1.0, 1.0])

### Visualization & Processing
- **`draw_on_image()`**: Overlays heatmaps on images with optional alpha blending
- **`draw()`**: Creates visualization with color mapping
- **`get_arr()`**: Returns the actual heatmap array in the specified value range
- **`get_arr_0to1()`**: Returns array in [0.0, 1.0] range

### Image Operations
- **`resize()`**: Resizes heatmaps to match image dimensions
- **`to_uint8()`**: Converts to uint8 format
- **`copy()`/`deepcopy()`**: Shallow and deep copying

### Key Design Choices
1. **Separation of concerns**: Stores data internally in [0.0, 1.0] range but can be converted to any value range
2. **Flexible input handling**: Accepts various input formats (uint8, float32)
3. **Image association**: Maintains reference to image shape for proper positioning
4. **Robust interpolation**: Uses cubic interpolation with clamping to prevent out-of-bounds values

This class is particularly useful for computer vision tasks involving attention maps, confidence heatmaps, or any spatial data visualization where heatmaps need to be overlaid on images.