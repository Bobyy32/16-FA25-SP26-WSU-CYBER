# Convert array to uint8 with clipping
result = change_dtype_(array, dtype=np.uint8, clip=True)

# Validate that dtype is uint8
allow_only_uint8(array)

# Gate dtypes with custom allowed/disallowed lists
gate_dtypes_strs(array, allowed="uint8 uint16", disallowed="float32")