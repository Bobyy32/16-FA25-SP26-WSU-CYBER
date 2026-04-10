import imgaug.augmenters as iaa

# Create a Jigsaw augmenter with a 3x3 grid
augmenter = iaa.Jigsaw(nb_rows=3, nb_cols=3, seed=42)

# Apply it
transformed_images = augmenter(images)