# Allow only uint8 dtype
from imgaug import augmenters as iaa

augmenter = iaa.OneOf(0.5)
augmenter.dtypes = ['uint8']

# Clip values between 0 and 255
array = gate_values(array, min_value=0, max_value=255)

# Validate dtype string against allowed list
allowed = "uint8"
disallowed = "uint16 uint32"