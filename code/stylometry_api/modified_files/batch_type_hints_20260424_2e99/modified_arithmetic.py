# ... code continues ...

    # Example: Using a tuple for random compression strength
    >>> aug = iaa.JpegCompression(compression=(10, 30))

    This randomly samples a compression strength between ``10`` and ``30`` per image.
    Higher values (closer to 100) mean stronger compression and lower quality.

    Note: ``JpegCompression`` is available as of imgaug version 0.4.0.

    See also :func:`~imgaug.augmenters.arithmetic.compress_jpeg` for the underlying
    function that performs the actual JPEG compression.

    """

    def __init__(self, compression=(0, 100),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(JpegCompression, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.compression = iap.handle_continuous_param(
            compression, "compression",
            value_range=(0, 100), tuple_to_uniform=True, list_to_choice=True)

    def _augment_batch_(self, batch, random_state, parents, hooks):
        # ... code as above ...