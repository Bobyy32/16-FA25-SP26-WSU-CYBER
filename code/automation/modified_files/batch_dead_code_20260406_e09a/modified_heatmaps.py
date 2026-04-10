import numpy as np


class HeatmapsOnImage:
    def __init__(self, arr, shape, min_value=0.0, max_value=1.0):
        self.arr = arr
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value
        self.arr_was_2d = False
        self.arr_is_2d = None

        # Process input array to normalize to [0.0, 1.0]
        if min_value != 0.0 or max_value != 1.0:
            # Adjust the array according to min_value, max_value
            arr_adjusted = self.change_normalization(arr, (min_value, max_value), (0.0, 1.0))
            self.arr = arr_adjusted
        else:
            self.arr = arr

    def get_arr(self):
        """Get the heatmap array with the correct value range.

        Returns
        ------
        ndarray
            (H,W) or (H,W,C) ndarray, float, usually [0.0, 1.0].
            When min_value != 0.0 and max_value != 1.0, returns an array
            with values in the interval [min_value, max_value].

        """
        arr = self.arr
        if not (self.min_value == 0.0 and self.max_value == 1.0):
            arr = self.change_normalization(arr, (self.min_value, self.max_value),
                                           (0.0, 1.0))
        return arr

    def to_uint8(self):
        """Convert this heatmaps object to an ``uint8`` array.

        Returns
        ------
        (H,W,C) ndarray
            Heatmap as an ``uint8`` array, i.e. with the discrete value
            range ``[0, 255]``.

        """
        # Convert from [0.0, 1.0] to [0, 255]
        arr_0to255 = np.clip(np.round(self.arr * 255), 0, 255)
        arr_uint8 = arr_0to255.astype(np.uint8)
        return arr_uint8

    @staticmethod
    def from_uint8(arr_uint8, shape, min_value=0.0, max_value=1.0):
        """Create a ``float``-based heatmaps object from an ``uint8`` array.

        Parameters
        ----------
        arr_uint8 : (H,W) ndarray or (H,W,C) ndarray
            Heatmap(s) array, where ``H`` is height, ``W`` is width
            and ``C`` is the number of heatmap channels.
            Expected dtype is ``uint8``.

        shape : tuple of int
            Shape of the image on which the heatmap(s) is/are placed.
            **Not** the shape of the heatmap(s) array, unless it is identical
            to the image shape (note the likely difference between the arrays
            in the number of channels).
            If there is not a corresponding image, use the shape of the
            heatmaps array.

        min_value : float, optional
            Minimum value of the float heatmaps that the input array
            represents. This will usually be 0.0. In most other cases it will
            be close to the interval ``[0.0, 1.0]``.
            Calling :func:`~imgaug.HeatmapsOnImage.get_arr`, will automatically
            convert the interval ``[0.0, 1.0]`` float array to this
            ``[min, max]`` interval.

        max_value : float, optional
            Maximum value of the float heatmaps that the input array
            represents. This will usually be 1.0.
            See parameter `min_value` for details.

        Returns
        ------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmaps object.

        """
        arr_0to1 = arr_uint8.astype(np.float32) / 255.0
        return HeatmapsOnImage.from_0to1(
            arr_0to1, shape,
            min_value=min_value,
            max_value=max_value)

    @staticmethod
    def from_0to1(arr_0to1, shape, min_value=0.0, max_value=1.0):
        """Create a heatmaps object from a ``[0.0, 1.0]`` float array.

        Parameters
        ----------
        arr_0to1 : (H,W) or (H,W,C) ndarray
            Heatmap(s) array, where ``H`` is the height, ``W`` is the width
            and ``C`` is the number of heatmap channels.
            Expected dtype is ``float32``.

        shape : tuple of ints
            Shape of the image on which the heatmap(s) is/are placed.
            **Not** the shape of the heatmap(s) array, unless it is identical
            to the image shape (note the likely difference between the arrays
            in the number of channels).
            If there is not a corresponding image, use the shape of the
            heatmaps array.

        min_value : float, optional
            Minimum value of the float heatmaps that the input array
            represents. This will usually be 0.0. In most other cases it will
            be close to the interval ``[0.0, 1.0]``.
            Calling :func:`~imgaug.HeatmapsOnImage.get_arr`, will automatically
            convert the interval ``[0.0, 1.0]`` float array to this
            ``[min, max]`` interval.

        max_value : float, optional
            Maximum value of the float heatmaps that the input array
            represents. This will usually be 1.0.
            See parameter `min_value` for details.

        Returns
        ------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmaps object.

        """
        # Initialize with default normalization
        heatmaps = HeatmapsOnImage(arr_0to1, shape,
                                    min_value=0.0, max_value=1.0)
        # Override min/max values if provided
        heatmaps.min_value = min_value
        heatmaps.max_value = max_value
        return heatmaps

    @classmethod
    def change_normalization(cls, arr, source, target):
        """Change the value range of a heatmap array.

        E.g. the value range may be changed from the interval ``[0.0, 1.0]``
        to ``[-1.0, 1.0]``.

        Parameters
        ----------
        arr : ndarray
            Heatmap array to modify.

        source : tuple of float
            Current value range of the input array, given as a
            tuple ``(min, max)``, where both are ``float`` values.

        target : tuple of float
            Desired output value range of the array, given as a
            tuple ``(min, max)``, where both are ``float`` values.

        Returns
        ------
        ndarray
            Input array, with new value range.

        """
        arr_min = source[0]
        arr_max = source[1]
        target_min = target[0]
        target_max = target[1]
        arr_new = arr * (target_max - target_min) / (arr_max - arr_min)
        arr_new = arr_new + target_min - arr_min * (target_max - target_min) / (arr_max - arr_min)
        return arr_new


def augment_heatmap_by_constant_value():
    """
    Augment heatmap by a constant value.
    """
    def random_value(min_value=-1, max_value=1):
        """
        Helper function to generate random values.
        """
        return np.random.uniform(min_value, max_value)

    def augment_by_constant_value(heatmap, factor, seed=None):
        """
        Augment a heatmap by a constant value.
        """
        return heatmap + factor

    return augment_by_constant_value