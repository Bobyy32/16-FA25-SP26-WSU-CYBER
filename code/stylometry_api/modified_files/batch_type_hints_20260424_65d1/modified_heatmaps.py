This code snippet is part of the `HeatmapsOnImage` class from the **imgaug** library. It handles the creation, conversion, and manipulation of heatmap objects, which are commonly used to represent probability or attention maps for image augmentation tasks.

## Key Methods and Features

### 1. **Heatmap Representation and Storage**
- The heatmap is stored internally as a float array in the `[0.0, 1.0]` interval.
- Additional attributes (`min_value`, `max_value`) allow custom value normalization.
- The class supports both 2D (`H × W`) and 3D (`H × W × C`) arrays.

### 2. **Conversion Functions**

#### `to_uint8()`
- Converts float heatmaps in `[0.0, 1.0]` to a uint8 array in `[0, 255]`.
- Performs rounding and clipping during conversion.
- Returns an array with the same height and width; channel count may be preserved.

#### `from_uint8()` (Static Method)
- Constructs a `HeatmapsOnImage` object from a uint8 array.
- Automatically scales the data to the `[0.0, 1.0]` range using the provided `min_value` and `max_value` parameters.

#### `from_0to1()` (Static Method)
- Constructs a `HeatmapsOnImage` object from a float array already normalized to `[0.0, 1.0]`.

### 3. **Normalization and Scaling**

#### `change_normalization()` (Class Method)
- Rescales a heatmap array to a new value range.
- Supports both tuple-based or `HeatmapsOnImage`-based input for `source` and `target` ranges.
- Optimizes the case where the source and target ranges are nearly identical to avoid unnecessary copying.

### 4. **Copy Methods**

#### `copy()` and `deepcopy()`
- Creates shallow and deep copies of the `HeatmapsOnImage` object, respectively.
- `deepcopy()` uses the `HeatmapsOnImage()` constructor with the current array and metadata, ensuring full data independence.

## Use Cases
- **Image Augmentation:** Heatmaps represent visual attention or classification probabilities.
- **Visualization:** Convert heatmaps to uint8 format for display in images.
- **Normalization Adjustment:** Allow users to rescale heatmaps for custom thresholds or display ranges.
- **Modularity and Portability:** Copy and deepcopy methods support clean manipulation and transferability of heatmap data without mutating the original object.

## Design Notes
- Heatmaps are designed to support both 2D and 3D input.
- The `shape` attribute refers to the target image dimensions, not necessarily the heatmap array dimensions.
- All static factory methods (`from_uint8`, `from_0to1`) accept `min_value` and `max_value` for flexible range control.

This implementation offers a flexible, extensible, and type-safe API for managing heatmaps within the imgaug library.