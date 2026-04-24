# Example logic (hypothetical usage)
if array.dtype != np.uint8:
    # Using allow_only_uint8
    gate_dtypes_strs(
        allow=['uint8'], 
        disallow=['float32', 'int32'],
        array=array 
    )