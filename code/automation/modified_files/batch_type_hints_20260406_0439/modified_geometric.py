import imgaug.augmenters as iaa

# Basic usage
aug = iaa.Jigsaw(nb_rows=5, nb_cols=5, max_steps=2)

# Use seed for determinism
aug = iaa.Jigsaw(seed=42)

# Deterministic mode
aug = iaa.Jigsaw(nb_rows=5, nb_cols=5, max_steps=2).to_deterministic(seed=42)

# Bounding Boxes and other CBAs are currently NOT supported