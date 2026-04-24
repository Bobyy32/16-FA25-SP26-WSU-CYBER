# Validate dtypes are uint8
allow_only_uint8(array, augmenter=my_augmentor)

# Validate against custom allowed/disallowed
gate_dtypes_strs(
    dtypes=[int64, float32],
    allowed="int8",
    disallowed="float64",
    augmenter=my_augmentor
)