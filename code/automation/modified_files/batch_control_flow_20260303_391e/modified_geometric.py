import imgaug.augmenters as iaa

aug = iaa.ElasticTransformation(alpha=(50, 100), sigma=12)
augmented_images = aug(augmenter_inputs={'images': images})