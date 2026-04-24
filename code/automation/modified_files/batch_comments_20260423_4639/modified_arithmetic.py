# Added in 0.4.0.
def get_parameters(self):
    """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
    return [self.compression]


# TODO: add more methods if needed

class JpegCompression(meta.Augmenter):
    """
    Degrade the quality of images by JPEG-compressing them.

    During JPEG compression, high frequency components (e.g. edges) are removed.
    With low compression (strength) only the highest frequency components are
    removed, while very high compression (strength) will lead to only the
    lowest frequency components "surviving". This lowers the image quality.
    For more details, see https://en.wikipedia.org/wiki/Compression_artifact.

    Note that this augmenter still returns images as numpy arrays (i.e. saves
    the images with JPEG compression and then reloads them into arrays). It
    does not return the raw JPEG file content.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.compress_jpeg`.

    Parameters
    ----------
    compression : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Degree of compression used during JPEG compression within value range
        ``[0, 100]``. Higher values denote stronger compression and will cause
        low-frequency components to disappear. Note that JPEG's compression
        strength is also often set as a *quality*, which is the inverse of this
        parameter. Common choices for the *quality* setting are around 80 to 95,
        depending on the image. This translates here to a *compression*
        parameter of around 20 to 5.

            * If a single number, then that value always will be used as the
              compression.
            * If a tuple ``(a, b)``, then the compression will be
              a value sampled uniformly from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and used as the compression.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing the
              compression for the ``n``-th image.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.JpegCompression(compression=(70, 99))

    Remove high frequency components in images via JPEG compression with
    a *compression strength* between ``70`` and ``99`` (randomly and
    uniformly sampled per image). This corresponds to a (very low) *quality*
    setting of ``1`` to ``30``.

    """

    def __init__(self, compression=(0, 100),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(JpegCompression, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        # will be converted to int during augmentation, which is why we allow
        # floats here
        self.compression = iap.handle_continuous_param(
            compression, "compression",
            value_range=(0, 100), tuple_to_uniform=True, list_to_choice=True)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        samples = self.compression.draw_samples((nb_images,),
                                                random_state=random_state)

        for i, (image, sample) in enumerate(zip(images, samples)):
            batch.images[i] = compress_jpeg(image, int(sample))

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.compression]