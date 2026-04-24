import imgaug.augmenters as iaa
aug = iaa.Jigsaw(nb_rows=10, nb_cols=10, allow_pad=True)
aug.augment_images(images) # Returns augmented image data