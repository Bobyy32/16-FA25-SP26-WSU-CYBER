import imgaug.augmenters as iaa

# Use gate_dtypes_strs to enforce allowed dtypes
iaa.Resize(validate_values=(0, 1))