This appears to be the implementation of the `HeatmapsOnImage` class from the `imgaug` library. I notice several important aspects and potential areas for improvement:

**Key Methods Present:**
- `to_0to1()` - Convert to 0-1 float range
- `to_uint8()` - Convert to uint8 array
- `from_uint8()` - Create from uint8 array
- `from_0to1()` - Create from 0-1 float array
- `change_normalization()` - Rescale value ranges
- `copy()` / `deepcopy()` - Copying methods

**Notable Observations:**

1. **`copy()` Implementation**: The docstring says "Shallow copy", but it calls `deepcopy()` - this is inconsistent.

2. **`to_uint8()` TODO**: It always returns (H,W,C) even if input was (H,W), which might need fixing.

3. **`change_normalization()`**: The validation logic for source/target parameters is in place, with float32 epsilon handling for equality checks.

4. **`from_0to1()`**: Sets default min/max values and then overwrites with passed parameters.

**Questions for you:**
- Are you looking to refactor or modify this code?
- Do you want to fix the `copy()` method inconsistency?
- Are you experiencing an issue with heatmaps conversion?
- Do you want to add additional methods (like `get_arr()` that's referenced in docs)?

Please let me know what you'd like to do with this code!