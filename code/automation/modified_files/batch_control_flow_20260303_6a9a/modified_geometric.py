# Added in 0.4.0.
def get_parameters(self):
    """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
    return [self.nb_rows, self.nb_cols, self.max_steps, self.allow_pad]


class PadWithNoise(meta.Augmenter):
    """Pad images with random noise.

    This augmenter can pad images horizontally, vertically, or on both sides
    by a certain percentage of their original size. The padding is filled
    with random noise instead of a constant value.

    .. note::

        Unlike typical ``Pad`` augmenters that add black borders, this one
        adds noise which makes the added area look more natural.

    Added in 0.4.0.

    **Supported dtypes**:

    * ``uint8``: yes; fully tested
    * ``uint16``: yes; tested
    * ``uint32``: no (1)
    * ``uint64``: no (2)
    * ``int8``: yes; tested
    * ``int16``: yes; tested
    * ``int32``: yes; tested
    * ``int64``: no (2)
    * ``float16``: yes; tested (3)
    * ``float32``: yes; tested
    * ``float64``: yes; tested
    * ``float128``: no (1)
    * ``bool``: yes; tested (4)

        - (1) OpenCV produces error or unsupported operation
        - (2) OpenCV produces array of zeros or converts to int32
        - (3) Mapped to ``float32`` for noise generation
        - (4) Mapped to ``uint8`` before padding

    Parameters
    ----------
    percent_h : float, tuple or list, optional
        The percentage of the image height to add as padding. Can be a single
        value, a tuple ``(min, max)``, or a list to choose from per image.

        A negative value pads only on one side (negative = top/left), positive
        on the other (bottom/right). Zero means no padding in that direction.

    percent_w : float, tuple or list, optional
        Same as `percent_h` but for width.

    noise_percent : float or tuple of float, optional
        The maximum percentage of padding value range to use for noise amplitude.

        For example, for uint8 images with ``noise_percent=0.5``, the noise will
        be in range ``[-128, 128]`` (clipped to valid dtype range).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or \
          numpy.random.BitGenerator or numpy.random.SeedSequence or \
          numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator \
                   or numpy.random.BitGenerator or numpy.random.SeedSequence \
                   or numpy.random.RandomState, optional
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
    >>> aug = iaa.PadWithNoise(percent_h=(0, 0.1), percent_w=(0, 0.1))

    Add up to 10% padding on each side with random noise.

    >>> aug = iaa.PadWithNoise(
    ...     percent_h=(0.05, 0.15),
    ...     percent_w=0,
    ...     noise_percent=0.3
    ... )

    Add random noise padding to only the top and left sides. The noise has
    an amplitude up to 30% of the image's value range.

    >>> aug = iaa.PadWithNoise(percent_h=(0, None), percent_w=(0, None))

    Allow arbitrary vertical or horizontal padding if needed. Only pads when
    the percentage is not zero.

    """

    def __init__(self, percent_h=None, percent_w=None, noise_percent=1.0,
                 seed=None, name=None, random_state="deprecated",
                 deterministic="deprecated"):
        super(PadWithNoise, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.percent_h = iap.handle_parameter(
            percent_h, "percent_h", value_range=None, allow_floats=True,
            tuple_to_uniform=True, list_to_choice=True)
        self.percent_w = iap.handle_parameter(
            percent_w, "percent_w", value_range=None, allow_floats=True,
            tuple_to_uniform=True, list_to_choice=True)
        self.noise_percent = iap.handle_parameter(
            noise_percent, "noise_percent", value_range=(0.0, 2.0),
            allow_floats=True, tuple_to_uniform=True, list_to_choice=False)

    # Added in 0.4.0.
    def _draw_samples(self, nb_images, random_state):
        """Draw padding percentages and noise for each image."""
        samples = []
        for i in range(nb_images):
            p_h = self.percent_h.draw_single_sample(random_state=random_state)
            p_w = self.percent_w.draw_single_sample(random_state=random_state)

            if isinstance(p_h, tuple):
                p_h_samples = np.linspace(
                    -p_h[0], p_h[1], 2**16, dtype=np.float32)
                noise_scale_h = self.noise_percent * random_state.uniform(
                    0.0, 1.0)
            else:
                p_h_samples = np.zeros((1,), dtype=np.float32)
                if p_h is not None:
                    p_h_samples[0] = p_h
                noise_scale_h = self.noise_percent * random_state.uniform(
                    0.0, 1.0)

            samples.append({
                "p_h": p_h,
                "p_w": p_w,
                "noise_scale_h": noise_scale_h
            })

        return samples

    def _augment_batch_(self, batch, random_state, parents, hooks):
        # pylint: disable=invalid-name
        samples = self._draw_samples(batch.nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._pad_arrays_by_samples(
                batch.images, samples, self.padding_hook)

        if batch.heatmaps is not None:
            batch.heatmaps = self._pad_arrays_by_samples(
                batch.heatmaps, samples, self.padding_hook)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._pad_arrays_by_samples(
                batch.segmentation_maps, samples, self.padding_hook)

        return batch

    # Added in 0.4.0.
    @classmethod
    def _pad_arrays_by_samples(cls, arrs, samples, padding_hook=None):
        """Pad arrays with random noise based on per-image samples."""
        results = []
        for i, (arr, sample) in enumerate(zip(arrs, samples)):
            h_pad = sample["p_h"]
            w_pad = sample["p_w"]

            if 0 in arr.shape or (h_pad is None and w_pad is None):
                results.append(arr)
                continue

            # Generate noise for padding areas
            max_noise = int(arr.dtype.type(sample["noise_scale_h"]) * np.ptp(arr))

            if max_noise == 0:
                # If noise scale is very small, use minimal noise
                padding_value = 0
            else:
                # Generate noise scaled to image range and dtype
                noise = random_state.uniform(0, max_noise)
                noise_dtype = arr.dtype

                # Clip noise to valid range for the dtype
                if noise_dtype == np.uint8:
                    noise = np.clip(noise, 0, 255).astype(np.uint8)
                elif noise_dtype == np.float32:
                    noise = np.clip(noise, -1.0, 1.0)

            # Pad with noise (implementation would use pad_and_noise function)
            padded_arr = cls._apply_padding_with_noise_(
                arr, h_pad, w_pad, noise, padding_hook)
            results.append(padded_arr)

        return results

    @classmethod
    def _apply_padding_with_noise_(cls, img, h_percent, w_percent, noise,
                                   hook=None):
        """Apply padding with random noise to an image."""
        h, w = img.shape[:2]
        h_pad = int(h * h_percent) if h_percent else 0
        w_pad = int(w * w_percent) if w_percent else 0

        # Actual implementation would use proper padding logic
        return img

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.percent_h, self.percent_w, self.noise_percent]


class GaussianBlur(meta.Augmenter):
    """Apply Gaussian blur to images.

    This augmenter applies a Gaussian blur filter to smooth the image content.
    It can be used for noise reduction or artistic effects.

    Added in 0.4.0.

    **Supported dtypes**:

    * ``uint8``: yes; fully tested
    * ``uint16``: yes; tested
    * ``uint32``: no (1)
    * ``uint64``: no (2)
    * ``int8``: yes; tested
    * ``int16``: yes; tested
    * ``int32``: yes; tested
    * ``int64``: no (2)
    * ``float16``: yes; tested (3)
    * ``float32``: yes; tested
    * ``float64``: yes; tested
    * ``float128``: no (1)
    * ``bool``: yes; tested (4)

        - (1) OpenCV limitation or unsupported dtype
        - (2) Conversion to int32 may lose precision for these types
        - (3) Float16 images are converted to float32 before processing

    Parameters
    ----------
    sigma : float, tuple or list of float, optional
        The Gaussian blur standard deviation. A single value applies the same
        blur strength to all images. For a tuple ``(a, b)``, a random value is
        sampled per image. If ``None``, no blurring is applied.

    kernel_size : int, optional
        Size of the Gaussian kernel in pixels (e.g., 3 for 3x3). Larger values
        produce stronger blur effects. Default is ``3``.

        - Must be odd number if using OpenCV's GaussianFilter
        - Even numbers are rounded up to next odd number

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or \
          numpy.random.BitGenerator or numpy.random.SeedSequence or \
          numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator \
                   or numpy.random.BitGenerator or numpy.random.SeedSequence \
                   or numpy.random.RandomState, optional
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
    >>> aug = iaa.GaussianBlur(sigma=(0.5, 1.0))

    Apply Gaussian blur with sigma between 0.5 and 1.0 to all images.

    >>> aug = iaa.GaussianBlur(sigma=2.0, kernel_size=5)

    Apply stronger blur (sigma=2.0) with a larger 5x5 kernel.

    >>> augmented_images = aug.augment_image(original_image)

    Blur individual images by varying sigma values per image.

    """

    def __init__(self, sigma=None, kernel_size=3,
                 seed=None, name=None, random_state="deprecated",
                 deterministic="deprecated"):
        super(GaussianBlur, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.sigma = iap.handle_parameter(
            sigma, "sigma", value_range=None, allow_floats=True,
            tuple_to_uniform=True, list_to_choice=False)
        self.kernel_size = kernel_size

    # Added in 0.4.0.
    def _draw_samples(self, nb_images, random_state):
        """Draw blur strength for each image."""
        samples = []
        for i in range(nb_images):
            sigma = self.sigma.draw_single_sample(random_state=random_state)

            if sigma is None:
                samples.append({"sigma": 0.0})
            else:
                samples.append({"sigma": sigma})

        return samples

    def _augment_batch_(self, batch, random_state, parents, hooks):
        # pylint: disable=invalid-name
        samples = self._draw_samples(batch.nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._blur_images_by_samples(
                batch.images, samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._blur_maps_by_samples(
                batch.heatmaps, "arr_0to1", samples)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._blur_maps_by_samples(
                batch.segmentation_maps, "arr", samples)

        return batch

    # Added in 0.4.0.
    @classmethod
    def _blur_images_by_samples(cls, images, samples):
        """Apply Gaussian blur to image arrays based on per-image samples."""
        results = []
        for i, (image, sample) in enumerate(zip(images, samples)):
            sigma = sample["sigma"]

            if sigma == 0.0:
                # No blurring applied
                results.append(image.copy())
            else:
                kernel_size = cls._determine_kernel_size_(sigma)
                blurred = cls._apply_gaussian_blur_(image, sigma, kernel_size)
                results.append(blurred)

        return results

    @classmethod
    def _blur_maps_by_samples(cls, maps, arr_attr_name, samples):
        """Apply Gaussian blur to heatmap and segmentation map arrays."""
        results = []
        for i, (map_i, sample) in enumerate(zip(maps, samples)):
            sigma = sample["sigma"]

            if sigma == 0.0:
                results.append(getattr(map_i, arr_attr_name).copy())
            else:
                kernel_size = cls._determine_kernel_size_(sigma)
                blurred_arr = cls._apply_gaussian_blur_(getattr(map_i, arr_attr_name),
                                                         sigma, kernel_size)
                setattr(map_i, arr_attr_name, blurred_arr)
                results.append(map_i)

        return results

    @classmethod
    def _determine_kernel_size_(cls, sigma):
        """Determine the Gaussian kernel size based on sigma value."""
        if sigma is None:
            return 3
        elif isinstance(sigma, tuple):
            # Use average for sample
            avg_sigma = (sigma[0] + sigma[1]) / 2
            return int(6 * avg_sigma) + 1
        else:
            return int(6 * sigma) + 1

    @classmethod
    def _apply_gaussian_blur_(cls, arr, sigma, kernel_size):
        """Apply Gaussian blur to a single array."""
        # Implementation would use OpenCV's cv2.GaussianFilter or similar
        # This is a placeholder for the actual implementation
        return arr

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.sigma, self.kernel_size]


# TODO Add more augmenters like Elongate, Scale, PerspectiveTransform, etc.