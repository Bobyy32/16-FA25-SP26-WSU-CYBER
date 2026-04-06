import imgaug.augmenters as iaa

# Basic usage
with iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.ElasticTransformation(alpha=(2, 4), sigma=6)
], random_state=0):
    augmented_images = images_augmented(images)