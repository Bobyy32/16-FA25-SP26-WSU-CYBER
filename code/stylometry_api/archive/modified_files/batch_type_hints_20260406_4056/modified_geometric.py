import imgaug.augmenters as iaa

# Create Jigsaw augmenter with fixed grid size
aug = iaa.Jigsaw(nb_rows=10, nb_cols=10)

# Apply augmentation
aug.augment_batch(batch, seed=42)