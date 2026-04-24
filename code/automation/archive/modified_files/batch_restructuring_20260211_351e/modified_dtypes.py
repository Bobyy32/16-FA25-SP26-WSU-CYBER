# Validate dtypes
gate_dtypes_strs(array, allowed="uint8", disallowed="float64")

# Clip to dtype range
clipped_array = clip_to_dtype_value_range_(array, dtype=np.uint8)

# Promote dtypes
promoted = promote_array_dtype(array, dtype=np.float64)