Based on the provided code snippet, here is a structured analysis of the **`KeypointsOnImage`** class from the `imgaug` library.

### 1. Class Purpose
`KeypointsOnImage` is a container object used to represent one or more keypoints (coordinate points) on a 2D image. It provides methods to manage, convert, and iterate over these keypoints, often in conjunction with distance maps.

### 2. Core Attributes
Although the `__init__` code isn't explicitly shown in the snippet, the class manages the following primary attributes:
*   **`keypoints`**: A list of `Keypoint` objects (instances with `x` and `y` coordinates).
*   **`shape`**: A tuple representing the height and width of the image on which the keypoints are located.
*   **`items`**: The underlying list of keypoints used for iteration.

### 3. Key Conversion Methods

#### `to_distance_maps(...)`
This method converts a list of keypoints into a 3D `distance_maps` array.
*   **Logic**:
    *   It creates a distance map for each keypoint where the value corresponds to the distance from that keypoint.
    *   **`inverted`**: Determines the search criteria. If `True`, the argmax is taken to find the keypoint (common for max-score maps); otherwise, `argmin` is used.
    *   **`threshold`**: Only maps with scores below the `threshold` are kept.
    *   **`if_not_found_coords`**: Coordinates to return if a keypoint cannot be found in the map.
    *   **`nb_channels`**: Specifies the number of channels in the output maps.
*   **Output**: A list of 3D numpy arrays (one for each keypoint).

#### `to_keypoints_on_image(...)`
Creates a new `KeypointsOnImage` instance from existing `keypoints` data.
*   **Logic**:
    *   Checks the list of keypoints.
    *   If no keypoints are found, returns an empty `KeypointsOnImage`.
    *   Otherwise, returns a new instance with the same data.

#### `invert_to_keypoints_on_image_()`
*   **Purpose**: Creates an inverted version of the keypoints on the image.
*   **Logic**:
    *   Inverts the coordinate system (e.g., `x` becomes `width - 1 - x`).
    *   Creates a new `KeypointsOnImage` with the inverted coordinates.

### 4. State Management & Duplication
The class supports several methods to manage the state of the keypoints:
*   **`copy`**: Returns a shallow copy of the `KeypointsOnImage`.
*   **`deepcopy`**: Returns a deep copy of the `KeypointsOnImage`.
*   **`__repr__`**: Generates a string representation showing the number of keypoints and their x/y coordinates.

### 5. Iteration and Access
The class implements standard sequence behavior:
*   **`__getitem__`**: Access individual keypoints by index.
*   **`__iter__`**: Iterate over the keypoints directly.
*   **`__len__`**: Returns the total count of keypoints.
*   **`__str__`**: Returns a formatted string representation (e.g., `"KeypointsOnImage(...)"`).

### 6. Usage Notes & Parameters
*   **Inversion Mode**: Used heavily in `to_distance_maps` when working with score maps where higher scores might indicate lower distances or vice versa.
*   **Thresholding**: Allows filtering of distance maps based on a minimum or maximum distance value (`threshold` parameter).
*   **Missing Data Handling**: The `if_not_found_coords` parameter is crucial for ensuring the output contains valid coordinates even if the keypoint is not explicitly present in the input map.

### Summary
The `KeypointsOnImage` class acts as the bridge between raw coordinate data and 2D distance/score maps. It provides robust tools to convert between these representations, handle missing data, and manage the state of keypoints, making it essential for computer vision tasks like pose estimation and object detection in `imgaug`.