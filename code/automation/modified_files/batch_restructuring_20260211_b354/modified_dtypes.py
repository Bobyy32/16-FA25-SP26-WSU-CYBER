# Validate dtypes
gate_dtypes_strs([img], allowed="uint8", disallowed="float64")

# Clip values to dtype range
clipped_img = clip_to_dtype_value_range_(img, np.uint8)

# Allow only uint8 (common for images)
allow_only_uint8([img])