import imgaug.augmenters as iaa

# Split images into a 10x10 grid and shuffle cells by default (0-2 steps)
aug = iaa.Jigsaw(nb_rows=10, nb_cols=10)

# Split into variable grids (1 to 4 rows/cols)
aug = iaa.Jigsaw(nb_rows=(1, 4), nb_cols=(1, 4))

# Set a random amount of steps between 1 and 10
aug = iaa.Jigsaw(nb_rows=1, nb_cols=1, max_steps=(1, 10))

# Make it deterministic for testing (recommended over the deprecated 'seed' arg)
aug = iaa.Jigsaw(nb_rows=10, nb_cols=10).to_deterministic(seed=42)