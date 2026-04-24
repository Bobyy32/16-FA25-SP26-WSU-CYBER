# Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.nb_rows, self.nb_cols, self.max_steps, self.allow_pad]


class Sequential(meta.Augmenter):
    """Apply one or multiple other augmenters sequentially.

    This is useful to compose multiple augmentation steps into a pipeline.
    You can create a complex augmentation flow by chaining augmenters
    together with this augmenter.

    Added in 0.4.0.

    **Supported dtypes**:

        Same as the supported dtypes of child augmenters (union of all).

    Parameters
    ----------
    children : list, optional
        One or more other imgaug.augmenters.meta.Augmenter instances to apply.
        If a single Augmenter is passed, it will be put into a list internally.
        Note that the order matters here! Augmentations are applied sequentially
        from left to right in the children list.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> # Basic sequential pipeline with multiple augmentations
    >>> aug = iaa.Sequential([
    >>>     iaa.FlipudOrFlip(),
    >>>     iaa.Affine(rotate=(-5, 5)),
    >>>     iaa.Multiply((0.9, 1.1))
    >>> ])

    You can also add augmentation with conditions:

    >>> aug = iaa.Sequential([
    >>>     iaa.SomeOf(2, [
    >>>         iaa.OneOf([
    >>>             iaa.FlipudOrFlip(),
    >>>             iaa.CropAndPad(percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    >>>         ]),
    >>>         iaa.RandomCropFromBorder(percent=(0.1, 0.3))
    >>>     ])
    >>> ])

    """

    def __init__(self, children=None,
                 random_state=None, name=None):
        super(Sequential, self).__init__(
            random_state=random_state, name=name)

        # Use list() to ensure we always have an iterable, even if None
        self.children = meta.handle_children_list(children, self.name, "then")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        # pylint: disable=invalid-name
        for child_idx, child in enumerate(self.children):
            batch = child.augment_batch_(batch,
                                         parents=parents + [child],
                                         hooks=hooks)

        return batch

    # Added in 0.4.0.
    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return self.children

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []

    # Added in 0.4.0.
    def _to_deterministic(self):
        augmented = self.copy()
        for child in augmented.children:
            child._to_deterministic()
        augmented.deterministic = True
        augmented.random_state = self.random_state.derive_rng_()
        return augmented

    # Added in 0.4.0.
    def __str__(self):
        pattern = (
            "%s("
            "children=%s, deterministic=%s"
            ")")
        return pattern % (self.__class__.__name__, self.children,
                         self.deterministic)


class SequentialPerChannel(meta.Augmenter):
    """Apply a different augmenter to each channel independently.

    This is similar to :class:`~imgaug.augmenters.Sequential`, but instead
    of applying the same augmentation sequence to all channels, it applies
    independent augmentations per channel.

    Added in 0.4.0.

    **Supported dtypes**:

        Same as the supported dtypes of child augmenters (union of all).

    Parameters
    ----------
    children : imgaug.augmenters.meta.Augmenter or list, optional
        One or more other imgaug.augmenters.meta.Augmenter instances to apply
        per channel. Each augmenter will be randomly selected from this list
        per channel and applied to each image's corresponding channel.

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
    >>> aug = iaa.SequentialPerChannel(
    >>>     children=[
    >>>         iaa.Multiply((0.9, 1.1)),
    >>>         iaa.GaussianNoise(var=(5, 20))
    >>>     ]
    >>> )

    Apply either multiplication or Gaussian noise to each channel independently.

    """

    def __init__(self, children=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(SequentialPerChannel, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        # Each channel gets a different augmenter selected from children
        self.children = meta.handle_children_list(children, self.name, "per_channel")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is not None:
            return self._augment_images(batch, random_state, parents, hooks)

        return batch

    # Added in 0.4.0.
    def _augment_images(self, batch, random_state, parents, hooks):
        augmented = []

        for child_idx, child in enumerate(self.children):
            row_children = []
            child_samples = child._draw_samples(batch.nb_rows, random_state)
            
            for image_idx, (image, child_sample) in enumerate(
                    zip(batch.images, child_samples)):
                # Get the augmented version of this child
                channel_augmenter = self._get_channel_augmenter(child_sample)
                
                if channel_augmenter is None:
                    continue

                row_children.append(channel_augmenter)

            batch.images[image_idx] = row_children[0].augment_batch_(
                batch.images[image_idx],
                random_state=random_state,
                parents=parents + [self],
                hooks=hooks
            )

        return augmented

    def _get_channel_augmenter(self, channel_sample):
        """Get the augmenter for a specific channel sample."""
        if isinstance(channel_sample, list):
            return self.children[channel_sample[0]]
        else:
            # Handle different cases for StochasticParameter resolution
            return None

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []

    # Added in 0.4.0.
    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return [self.children]

    # Added in 0.4.0.
    def _to_deterministic(self):
        augmented = self.copy()
        for child in augmented.children:
            child._to_deterministic()
        augmented.deterministic = True
        augmented.random_state = self.random_state.derive_rng_()
        return augmented

    # Added in 0.4.0.
    def __str__(self):
        pattern = (
            "%s("
            "children=%s, deterministic=%s"
            ")")
        return pattern % (self.__class__.__name__, self.children,
                         self.deterministic)


# End of imgaug/augmenters/meta.py content
# Additional augmenters would follow in separate files or sections:

# __all__ = [
#     "Augmenter",
#     "Sequential",
#     "SequentialPerChannel",
#     # ... other augmenters from other modules
# ]


class ImageAugmentation(object):
    """Base class for all image augmentation augmenters.

    This abstract base class provides common functionality and interfaces
    for all augmenters in imgaug. Subclasses should implement their specific
    augmentation behavior.

    Added in 0.4.0.

    Parameters
    ----------
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

    """

    def __init__(self, seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        self.seed = seed
        self.name = name
        self.random_state = _default_rng(seed)
        if random_state != "deprecated":
            # Handle the deprecated parameter
            pass
        if deterministic != "deprecated":
            # Handle the deprecated parameter
            self.deterministic = deterministic

    def augment_image(self, image, dtype_in=np.uint8):
        """Augment a single image.

        Parameters
        ----------
        image : array-like
            The input image to be augmented. Expected shape is (H, W) or (H, W, C).

        dtype_in : np.dtype or None, optional
            Input dtype of the image. If None, infers from image data type.

        Returns
        -------
        array-like
            Augmented image with same shape as input.
        """
        raise NotImplementedError(
            "Subclasses must implement augment_image() method.")

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []

    # Added in 0.4.0.
    def to_deterministic(self, random_state=None):
        """Convert to deterministic mode.

        Returns a copy of the augmenter with deterministic behavior and
        a fixed random state derived from `random_state`.

        Parameters
        ----------
        random_state : None or int, optional
            Seed for deterministic randomness. If None (default), uses current
            timestamp.

        Returns
        -------
        Augmenter
            A new augmenter instance in deterministic mode.

        """
        raise NotImplementedError(
            "Subclasses must implement to_deterministic() method.")


class _DeprecatedBase(object):
    """Base class for deprecated augmenters or helpers."""
    # Added in 0.5.0 to handle legacy code paths
    def __init__(self):
        super(_DeprecatedBase, self).__init__()

    # Added in 0.5.0
    def _deprecated_helper(self, arg=None):
        if arg is None:
            raise ValueError("Argument 'arg' required for deprecated helper.")
        return arg


class ColorJitter(meta.Augmenter):
    """Adjust colors by changing brightness and contrast.

    This augmenter modifies the color values of images by adding noise or
    changing the intensity range to increase variability in color appearance.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``int16``: no (deprecated)
        * ``int32``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``bool``: yes; fully tested

    Parameters
    ----------
    brightness_factor : imgaug.parameters.StochasticParameter, optional
        Factor by which to multiply the image brightness. Can be a float value
        or a tuple of floats for random sampling. Values can range from -0.5
        to +0.5 where 0 means no change.

    contrast_factor : imgaug.parameters.StochasticParameter, optional
        Factor by which to multiply the image contrast. Similar behavior as
        brightness_factor but applied differently.

    """

    def __init__(self, brightness_factor=None, contrast_factor=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ColorJitter, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        if brightness_factor is not None:
            self.brightness_factor = iap.handle_stochastic_param(
                brightness_factor, "brightness_factor", allow_floats=True)
        else:
            # Default value in later versions uses a small range around zero
            self.brightness_factor = iap.StochasticParameter(
                uniform=(-0.2, 0.2), name="brightness")

        if contrast_factor is not None:
            self.contrast_factor = iap.handle_stochastic_param(
                contrast_factor, "contrast_factor", allow_floats=True)
        else:
            self.contrast_factor = iap.StochasticParameter(
                uniform=(-0.2, 0.2), name="contrast")

    def _draw_samples(self, nb_images, random_state):
        return self.brightness_factor.draw_samples((nb_images,),
                                                   random_state=random_state), \
               self.contrast_factor.draw_samples((nb_images,),
                                                 random_state=random_state)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        b_factors, c_factors = self._draw_samples(batch.nb_rows, random_state)

        for i, (b, c) in enumerate(zip(b_factors, c_factors)):
            if batch.images is not None:
                batch.images[i] = self._apply_brightness_contrast(
                    batch.images[i], b, c)

        return batch

    # Added in 0.4.0.
    def _apply_brightness_contrast(self, image, brightness_factor,
                                   contrast_factor):
        h, w, ch = image.shape
        
        if brightness_factor != 0 or contrast_factor != 0:
            # Apply the transforms
            result = np.zeros((h, w, ch), dtype=image.dtype)

            for c_idx in np.arange(ch):
                channel = image[..., c_idx]
                
                # Calculate adjusted brightness and contrast
                brightness = brightness_factor * h / (255.0 if image.max() > 128 else 1)
                contrast = contrast_factor * ch
                
                result[..., c_idx] = np.clip(
                    channel + brightness, 0, 255 if image.dtype == np.uint8 else None
                )

        return result

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.brightness_factor, self.contrast_factor]

    # Added in 0.4.0.
    def _to_deterministic(self):
        augmented = self.copy()
        augmented.deterministic = True
        augmented.random_state = self.random_state.derive_rng_()
        return augmented

    # Added in 0.4.0.
    def __str__(self):
        pattern = (
            "%s("
            "brightness_factor=%s, contrast_factor=%s, deterministic=%s"
            ")")
        return pattern % (self.__class__.__name__, self.brightness_factor,
                         self.contrast_factor, self.deterministic)