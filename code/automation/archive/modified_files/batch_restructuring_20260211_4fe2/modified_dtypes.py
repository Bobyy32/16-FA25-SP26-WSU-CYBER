# Validate dtypes
imgaug.dtypes.gate_dtypes_strs(
    array, 
    allowed="uint8 uint16", 
    disallowed="float32 float64"
)

# Clip values to dtype range
clipped_array = imgaug.dtypes.clip_to_dtype_value_range_(array, np.uint8)

# Promote to common dtype
result_dtype = imgaug.dtypes.promote_dtypes([np.float32, np.int32])