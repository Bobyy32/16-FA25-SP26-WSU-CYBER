import imgaug.augmenters as iaa
import imgaug.core as iaa

# Create a Jigsaw augmenter with fixed 3x5 grid
aug = iaa.Jigsaw(nb_rows=3, nb_cols=5, max_steps=1)

# Use it in a pipeline
imgaug_pipeline = iaa.Sequential([
    iaa.GaussianBlur(2),
    iaa.Jigsaw(nb_rows=3, nb_cols=5, max_steps=1),
    iaa.FlipHorizontal()
])

# Augment data
augmented_image = aug.augment_image(original_image)