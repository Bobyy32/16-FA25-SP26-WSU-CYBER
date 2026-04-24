import imgaug as ia

# 1. Load a heatmap as a float array in [0, 1]
arr_float = ... # shape (H, W, C)

# 2. Convert to int8 (for saving or display)
uint8_arr = arr_float.to_uint8()

# 3. Load back or create from int8
hm_uint8 = HeatmapsOnImage.from_uint8(uint8_arr, shape=(100, 100))

# 4. Modify normalization range if needed (e.g., to [-1, 1])
hm_float_01 = hm_uint8.to_0to1() # (Assuming method exists, likely via get_arr/convert)
# Note: 'from_0to1' is static, so to get a HeatmapsOnImage from float:
# hm_float = HeatmapsOnImage.from_0to1(arr_float, shape=(100, 100), min=-1.0, max=1.0)
# Change range logic:
target_arr = HeatmapsOnImage.change_normalization(arr_float, (0.0, 1.0), (-1.0, 1.0))