# Ensure the image is uint8 and values are within [0, 255]
from imgaug import augmenters as iaa

# This would internally trigger gate_dtypes_strs validation
augmenter = iaa.Resize(allowed_dtypes="uint8") 
# Or explicitly via a meta-augmenter
augmenter = iaa.SomeAugmenter(validate_output=True)
augmenter.validate_output(...) # Would call _gate_dtypes and _gate_value