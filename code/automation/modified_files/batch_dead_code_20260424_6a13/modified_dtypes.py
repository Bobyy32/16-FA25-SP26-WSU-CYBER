import numpy as np
import imgaug as ia

# Convert to dtype strings
allowed = "uint8"
disallowed = "uint16 uint32"

# Validate
try:
    gate_dtypes_strs(
        dtypes=[np.dtype("uint8"), np.dtype("float64")],
        allowed=allowed,
        disallowed=disallowed,
        augmenter=ia.augmenters.meta.RandomFlip()
    )
except ValueError as e:
    print("Error:", e)