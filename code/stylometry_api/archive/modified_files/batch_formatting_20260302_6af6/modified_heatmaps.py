This code defines a `HeatmapsOnImage` class for handling heatmaps in image processing tasks. Here's a breakdown of its key features:

## Main Purpose
The class represents heatmaps that are "on" an image, storing both the heatmap data and the image's shape information.

## Key Methods

### **Initialization & Creation**
- `__init__()`: Sets up the heatmap with array data and image shape
- `from_uint8()`: Creates heatmaps from uint8 arrays (0-255 range)
- `from_0to1()`: Creates heatmaps from float arrays in [0.0, 1.0] range

### **Data Conversion**
- `to_uint8()`: Converts internal [0.0, 1.0] float array to uint8 (0-255)
- `change_normalization()`: Transforms value ranges (e.g., [0,1] → [-1,1])

### **Visualization & Display**
- `draw()`: Renders heatmaps as colored overlays on images
- `draw_on_image()`: Draws heatmaps directly on image arrays
- `get_arr()`: Returns the underlying heatmap array with proper value range

### **Image Operations**
- `resize()`: Scales heatmaps to different sizes
- `pad()`: Adds padding around heatmaps
- `crop()`: Extracts portions of heatmaps

### **Utility Functions**
- `copy()`/`deepcopy()`: Creates copies of heatmap objects
- `get_shape()`: Returns the image shape
- `get_arr()`: Returns the actual heatmap data array

## Key Features
1. **Value Range Management**: Stores min/max values to properly convert between [0,1] and other ranges
2. **Flexible Input**: Accepts different data types (uint8, float32)
3. **Image Integration**: Maintains relationship with underlying images
4. **Visualization**: Can draw heatmaps as colored overlays
5. **Memory Efficient**: Supports shallow/deep copying

## Use Cases
- Object detection heatmaps
- Semantic segmentation
- Attention maps
- Feature visualization
- Medical imaging heatmaps

The class is designed to be integrated into image processing pipelines where heatmaps need to be visualized, transformed, or analyzed in conjunction with images.