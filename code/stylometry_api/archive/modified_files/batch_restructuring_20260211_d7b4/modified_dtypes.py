# Validate that inputs are uint8 only
gate_dtypes_strs(images, allowed="uint8", disallowed="*")

# Allow specific dtypes, warn about others
gate_dtypes_strs(images, allowed="uint8 float32", disallowed="uint64")

# Clip values to valid range for a dtype
clipped = clip_to_dtype_value_range_(array, np.uint8)