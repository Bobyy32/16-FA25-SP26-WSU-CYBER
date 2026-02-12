# Validate dtypes
imgaug.dtypes.gate_dtypes_strs(
    array, 
    allowed="uint8 uint16", 
    disallowed="float64"
)

# Clip values to dtype range
clipped_array = imgaug.dtypes.clip_to_dtype_value_range_(array, np.uint8)

# Ensure compatible dtypes
result = imgaug.dtypes.promote_array_dtypes([array1, array2])