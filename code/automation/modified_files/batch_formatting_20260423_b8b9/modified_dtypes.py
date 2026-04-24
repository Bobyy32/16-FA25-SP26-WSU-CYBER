from imgaug import augmenters as iaa

augmenter = iaa.OneOf([
    iaa.Gaussian(sigma=(0, 2)),
    iaa.Translation((0, 50)),
])

image = image.astype(np.uint8)  # Must be uint8 for this augmenter
augmenter(image)  # Validates and warns if types are not permitted