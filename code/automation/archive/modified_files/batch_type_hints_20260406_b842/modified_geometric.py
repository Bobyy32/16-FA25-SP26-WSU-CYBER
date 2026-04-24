import imgaug.augmenters as iaa

# 1. Create the augmenter
jigsaw = iaa.augmenters.Jigsaw(
    nb_rows=3,
    nb_cols=3,
    max_steps=10,
    allow_pad=False,
    deterministic=True # Recommended instead of seed
)

# 2. Apply to image
# augmented = jigsaw(image)

# 3. Apply to batch
# augmented_batch = jigsaw(batch_images)