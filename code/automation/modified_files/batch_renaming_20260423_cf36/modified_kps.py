This code snippet is from the **imgaug** library, specifically the `KeypointsOnImage` class found in the module `imgaug.augmentables.kps`. It defines a container class for managing keypoints on an image, providing utilities for conversion, copying, iteration, and coordinate management.

### Class Overview
`KeypointsOnImage` represents a set of 2D or 3D keypoints associated with a specific image shape. It acts similarly to other augmentables like `BoundingBoxesOnImage` or `GeneralizedKeyPoints`, offering standardized methods for data manipulation.

### Key Methods & Functionality
1. **`to_distance_maps(...)`**:
   - Converts keypoints into a distance map representation. This is useful for tasks like template matching or distance-based augmentations.
   - Supports both **inverted** and **non-inverted** modes, allowing flexibility in distance calculations.
   - Handles keypoints not found using the `if_not_found_coords` parameter to allow flexible search behavior.

2. **`to_keypoints_on_image()`**:
   - Returns a new `KeypointsOnImage` instance containing the same keypoints as the current one.
   - Acts as a copy method that preserves the keypoints' structure and values.

3. **`invert_to_keypoints_on_image()`**:
   - Retrieves keypoints from another `KeypointsOnImage` instance while applying the inversion transformation to the coordinates.

4. **`copy()` and `deepcopy()`**:
   - These methods allow users to create shallow or deep copies of the keypoints object, respectively, ensuring data isolation or complete duplication.

5. **Container Methods**:
   - Includes standard container methods like `__getitem__`, `__iter__`, and `__len__`, allowing easy access to individual keypoints or iteration over the keypoints collection.

### Important Features
- **Shape Consistency**: All methods are aware of the image shape and ensure keypoints align with the correct dimensions.
- **Threshold Handling**: Methods like `to_distance_maps` use a `threshold` parameter to filter out keypoints beyond a certain distance.
- **Performance**: The code is optimized for performance, with specific handling for large datasets (e.g., using `copy` instead of `deepcopy` for some operations).
- **Extensibility**: By supporting various parameter configurations (like `drop_if_not_found`, `min_distance`, etc.), it offers high flexibility for augmentation workflows.

### Summary
This snippet demonstrates the core functionality of the `KeypointsOnImage` class, showing how it handles keypoints with image-aligned data structures, conversion utilities, and efficient copy operations. The design supports both simple access and advanced manipulation, making it a powerful tool for tasks like object detection and augmentation.