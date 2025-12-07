"""
Augmenters that apply changes to images based on segmentation methods.

List of augmenters:
    * :class:`Superpixels`
    * :class:`Voronoi`
    * :class:`UniformVoronoi`
    * :class:`RegularGridVoronoi`
    * :class:`RelativeRegularGridVoronoi`
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Standard library imports
from __future__ import print_function, division, absolute_import
from abc import ABCMeta, abstractmethod

# Third-party library imports
import numpy as np
import six
import six.moves as sm
# Note: use skimage.segmentation instead of 'from skimage import segmentation'
# to avoid unittest mixing up imgaug.augmenters.segmentation with skimage.segmentation
import skimage.segmentation
import skimage.measure

# Local/internal imports
import imgaug as ia
from . import meta
from .. import random as iarandom
from .. import parameters as iap
from .. import dtypes as iadt
from ..imgaug import _NUMBA_INSTALLED, _numbajit


# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

_SLIC_SUPPORTS_START_LABEL = (
    tuple(map(int, skimage.__version__.split(".")[0:2]))
    >= (0, 17)
)  # Added in 0.5.0.


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _ensure_image_max_size(image, max_size, interpolation):
    """Ensure that images do not exceed a required maximum sidelength.

    This downscales to `max_size` if any side violates that maximum.
    The other side is downscaled too so that the aspect ratio is maintained.

    **Supported dtypes**:
    See :func:`~imgaug.imgaug.imresize_single_image`.

    Parameters
    ----------
    image : ndarray
        Image to potentially downscale.

    max_size : int
        Maximum length of any side of the image.

    interpolation : string or int
        See :func:`~imgaug.imgaug.imresize_single_image`.
    """
    if max_size is not None:
        size = max(image.shape[0], image.shape[1])
        if size > max_size:
            resize_factor = max_size / size
            new_height = int(image.shape[0] * resize_factor)
            new_width = int(image.shape[1] * resize_factor)
            image = ia.imresize_single_image(
                image,
                (new_height, new_width),
                interpolation=interpolation)
    return image


def _verify_sample_points_images(images):
    """Verify that sample points images are valid."""
    assert len(images) > 0, (
        "Cannot sample points on zero images.")
    for i, image in enumerate(images):
        assert image.shape[0] > 0 and image.shape[1] > 0, (
            "Expected image %d to have height and width greater than zero, "
            "got shape %s." % (i, image.shape,))


def _blur_mean_shift(image, spatial_window_radius, color_window_radius):
    """Apply mean shift filtering."""
    if spatial_window_radius is not None or color_window_radius is not None:
        from scipy import ndimage

        # Apply mean shift filtering if requested
        # pylint falsely thinks that ndimage is the numpy.ndarray and
        # hence doesn't have a filters attribute
        # pylint: disable=no-member
        if spatial_window_radius is not None:
            # note that while the function is called "median", in
            # combination with the mean kernel this should be
            # equivalent to mean shift filtering
            image = ndimage.filters.generic_filter(
                image,
                function=np.mean,
                size=(spatial_window_radius, spatial_window_radius, 0),
                mode="mirror"
            )

        if color_window_radius is not None:
            channel_axis = 2
            for c in sm.xrange(image.shape[channel_axis]):
                # note that while the function is called "median", in
                # combination with the mean kernel this should be
                # equivalent to mean shift filtering
                image[:, :, c] = ndimage.filters.generic_filter(
                    image[:, :, c],
                    function=np.mean,
                    size=color_window_radius,
                    mode="mirror"
                )
    return image


# ==============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# ==============================================================================

if _NUMBA_INSTALLED:
    # Placed here so that the decorators are only executed if numba is
    # actually installed. This also allows to test them once per file.
    @_numbajit("int32[:, :](int32[:, :], int32[:], int32[:], float64[:,:])", cache=True)
    def _draw_voronoi_cells_from_labels_jit_int32(cell_image, xx, yy, rel_seed_coord):
        # int64 version is below
        height, width = cell_image.shape
        for i_point in range(len(xx)):
            x, y = xx[i_point], yy[i_point]
            if 0 <= x < width and 0 <= y < height:
                xx_p, yy_p = x, y
                if len(rel_seed_coord) > 0:
                    dists_calc = (rel_seed_coord[:, 0] - x) ** 2 + (rel_seed_coord[:, 1] - y) ** 2
                    label = np.int32(np.argmin(dists_calc))
                else:
                    label = np.int32(1)
                cell_image[y, x] = label
        return cell_image

    @_numbajit("int64[:, :](int64[:, :], int64[:], int64[:], float64[:, :])", cache=True)
    def _draw_voronoi_cells_from_labels_jit_int64(cell_image, xx, yy, rel_seed_coord):
        # int32 version is above
        height, width = cell_image.shape
        for i_point in range(len(xx)):
            x, y = xx[i_point], yy[i_point]
            if 0 <= x < width and 0 <= y < height:
                xx_p, yy_p = x, y
                if len(rel_seed_coord) > 0:
                    dists_calc = (rel_seed_coord[:, 0] - x) ** 2 + (rel_seed_coord[:, 1] - y) ** 2
                    label = np.int64(np.argmin(dists_calc))
                else:
                    label = np.int64(1)
                cell_image[y, x] = label
        return cell_image

    @_numbajit("float64[:, :, :](float64[:, :, :], int32[:, :, :])", cache=True)
    def _replace_segments_numba_float64(input_dtype, replacements):
        output_dtype = np.copy(input_dtype)
        height, width = output_dtype.shape[0:2]
        for y in range(height):
            for x in range(width):
                if replacements[y, x, 0] == 1:
                    output_dtype[y, x, 0] = replacements[y, x, 1]
                    output_dtype[y, x, 1] = replacements[y, x, 2]
                    output_dtype[y, x, 2] = replacements[y, x, 3]
        return output_dtype

    @_numbajit("uint8[:, :, :](uint8[:, :, :], int32[:, :, :])", cache=True)
    def _replace_segments_numba_uint8(input_dtype, replacements):
        output_dtype = np.copy(input_dtype)
        height, width = output_dtype.shape[0:2]
        for y in range(height):
            for x in range(width):
                if replacements[y, x, 0] == 1:
                    output_dtype[y, x, 0] = replacements[y, x, 1]
                    output_dtype[y, x, 1] = replacements[y, x, 2]
                    output_dtype[y, x, 2] = replacements[y, x, 3]
        return output_dtype

    @_numbajit("int32[:, :, :](int64[:], int32[:, :], float64[:, :], float64[:], int32, int32, int32)", cache=True)
    def _compute_replacements_numba(replace_samples, segmentation_map, segment_means, p_replace_samples, height, width, nb_channels):
        replacements = np.zeros((height, width, 1+nb_channels), dtype=np.int32)
        for i, p_replace_sample in enumerate(p_replace_samples):
            if replace_samples[i] == 1:
                for y in range(height):
                    for x in range(width):
                        if segmentation_map[y, x] == i:
                            replacements[y, x, 0] = 1
                            for c in range(nb_channels):
                                replacements[y, x, 1+c] = int(segment_means[i, c])
        return replacements
else:
    # Dummy functions when numba is not installed
    def _draw_voronoi_cells_from_labels_jit_int32(*args):
        raise NotImplementedError("Numba is required for this function")
    
    def _draw_voronoi_cells_from_labels_jit_int64(*args):
        raise NotImplementedError("Numba is required for this function")
    
    def _replace_segments_numba_float64(*args):
        raise NotImplementedError("Numba is required for this function")
    
    def _replace_segments_numba_uint8(*args):
        raise NotImplementedError("Numba is required for this function")
    
    def _compute_replacements_numba(*args):
        raise NotImplementedError("Numba is required for this function")


# ==============================================================================
# MAIN AUGMENTER CLASSES
# ==============================================================================

class Superpixels(meta.Augmenter):
    """Transform images partially/completely to their superpixel representation.

    This implementation uses skimage's version of the SLIC algorithm.

    .. note::
        This augmenter is fairly slow. See :ref:`performance`.

    **Supported dtypes**:

    if (image size <= max_size):
        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: limited (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: limited (1)
        * ``float16``: no (2)
        * ``float32``: no (2)
        * ``float64``: no (3)
        * ``float128``: no (2)
        * ``bool``: yes; tested

        - (1) Superpixel mean intensity replacement requires computing
              these means as ``float64`` s. This can cause inaccuracies for
              large integer values.
        - (2) Error in scikit-image.
        - (3) Loss of resolution in scikit-image.

    if (image size > max_size):
        minimum of (
            ``imgaug.augmenters.segmentation.Superpixels(image size <= max_size)``,
            :func:`~imgaug.augmenters.segmentation._ensure_image_max_size`
        )

    Parameters
    ----------
    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        
    n_segments : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Rough target number of how many superpixels to generate.
        
    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        
    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is exceeded.
        
    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.
        
    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.
        
    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        
    deterministic : bool, optional
        Deprecated since 0.4.0.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Superpixels(p_replace=1.0, n_segments=64)
    """

    def __init__(self, p_replace=0.5, n_segments=100, max_size=128,
                 interpolation="linear", seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Superpixels, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.p_replace = iap.handle_probability_param(p_replace, "p_replace")
        self.n_segments = iap.handle_discrete_param(
            n_segments, "n_segments", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.max_size = max_size
        self.interpolation = interpolation

    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        rss = random_state.duplicate(len(images))

        for i, (image, rs) in enumerate(zip(images, rss)):
            batch.images[i] = self._augment_single_image(image, rs)

        return batch

    def _augment_single_image(self, image, random_state):
        n_segments = self.n_segments.draw_sample(random_state)
        
        image_sp = self._reseed(image, n_segments, random_state)
        
        if self.p_replace.a == 1.0 and self.p_replace.b == 1.0:
            image = image_sp
        else:
            p_replace_samples = self.p_replace.draw_samples(
                (n_segments,), random_state)
            mask = (p_replace_samples > 0.5).astype(np.uint8)
            image = self._apply_segmentation_mask(image, image_sp, mask)
        
        return image

    def _reseed(self, image, n_segments, random_state):
        # Implementation of superpixel segmentation
        # This is a simplified version - actual implementation would be more complex
        pass

    def _apply_segmentation_mask(self, image_orig, image_segmented, mask):
        # Apply mask to blend original and segmented images
        # This is a simplified version - actual implementation would be more complex
        pass

    def __repr__(self):
        return "Superpixels(p_replace=%s, n_segments=%s, max_size=%s, interpolation=%s)" % (
            self.p_replace, self.n_segments, self.max_size, self.interpolation)

    def __str__(self):
        return self.__repr__()


class _AbstractVoronoi(six.with_metaclass(ABCMeta, meta.Augmenter)):
    """Abstract base class for Voronoi-based augmenters."""

    def __init__(self, points_sampler, p_drop_points=None,
                 p_replace=1.0, max_size=512,
                 interpolation="linear", seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(_AbstractVoronoi, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        
        self.points_sampler = points_sampler
        self.p_drop_points = iap.handle_probability_param(
            p_drop_points, "p_drop_points") if p_drop_points is not None else None
        self.p_replace = iap.handle_probability_param(p_replace, "p_replace")
        self.max_size = max_size
        self.interpolation = interpolation

    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        rss = random_state.duplicate(nb_images)

        for i, (image, rs) in enumerate(zip(images, rss)):
            batch.images[i] = self._augment_single_image(image, rs)

        return batch

    @abstractmethod
    def _augment_single_image(self, image, random_state):
        """Augment a single image."""
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return self.__repr__()


class Voronoi(_AbstractVoronoi):
    """Uniformly sample Voronoi cells on images and recolor them."""

    def __init__(self, points_sampler, p_drop_points=None, p_replace=1.0,
                 max_size=128, interpolation="linear", seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Voronoi, self).__init__(
            points_sampler=points_sampler, p_drop_points=p_drop_points,
            p_replace=p_replace, max_size=max_size,
            interpolation=interpolation, seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    def _augment_single_image(self, image, random_state):
        # Implementation of Voronoi augmentation
        # This is a simplified version - actual implementation would be more complex
        pass

    def __repr__(self):
        return "Voronoi(%s, p_drop_points=%s, p_replace=%s, max_size=%s, interpolation=%s)" % (
            self.points_sampler, self.p_drop_points, self.p_replace,
            self.max_size, self.interpolation)


class UniformVoronoi(Voronoi):
    """Sample Voronoi cells uniformly on images and recolor them."""

    def __init__(self, n_points=(50, 500), p_drop_points=None,
                 p_replace=1.0, max_size=128, interpolation="linear",
                 seed=None, name=None, random_state="deprecated",
                 deterministic="deprecated"):
        # Create UniformPointsSampler
        from imgaug.augmenters.segmentation import UniformPointsSampler
        points_sampler = UniformPointsSampler(n_points)
        
        super(UniformVoronoi, self).__init__(
            points_sampler=points_sampler, p_drop_points=p_drop_points,
            p_replace=p_replace, max_size=max_size,
            interpolation=interpolation, seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class RegularGridVoronoi(Voronoi):
    """Sample Voronoi cells on regular grids and recolor them."""

    def __init__(self, n_rows=(10, 30), n_cols=(10, 30),
                 p_drop_points=0.4, p_replace=1.0,
                 max_size=128, interpolation="linear",
                 seed=None, name=None, random_state="deprecated",
                 deterministic="deprecated"):
        # Create RegularGridPointsSampler
        from imgaug.augmenters.segmentation import RegularGridPointsSampler
        points_sampler = RegularGridPointsSampler(n_rows, n_cols)
        
        super(RegularGridVoronoi, self).__init__(
            points_sampler=points_sampler, p_drop_points=p_drop_points,
            p_replace=p_replace, max_size=max_size,
            interpolation=interpolation, seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class RelativeRegularGridVoronoi(Voronoi):
    """Sample Voronoi cells on relative regular grids and recolor them."""

    def __init__(self, n_rows_frac=(0.05, 0.15), n_cols_frac=(0.05, 0.15),
                 p_drop_points=0.4, p_replace=1.0,
                 max_size=128, interpolation="linear",
                 seed=None, name=None, random_state="deprecated",
                 deterministic="deprecated"):
        # Create RelativeRegularGridPointsSampler
        from imgaug.augmenters.segmentation import RelativeRegularGridPointsSampler
        points_sampler = RelativeRegularGridPointsSampler(n_rows_frac, n_cols_frac)
        
        super(RelativeRegularGridVoronoi, self).__init__(
            points_sampler=points_sampler, p_drop_points=p_drop_points,
            p_replace=p_replace, max_size=max_size,
            interpolation=interpolation, seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# ==============================================================================
# POINT SAMPLER CLASSES
# ==============================================================================

class IPointsSampler(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for point samplers."""

    @abstractmethod
    def sample_points(self, images, random_state):
        """Sample points on images."""
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class RegularGridPointsSampler(IPointsSampler):
    """Sample points on regular grids."""

    def __init__(self, n_rows, n_cols):
        self.n_rows = iap.handle_discrete_param(
            n_rows, "n_rows", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.n_cols = iap.handle_discrete_param(
            n_cols, "n_cols", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(len(images))
        points = []
        
        for image, rs in zip(images, rss):
            n_rows = self.n_rows.draw_sample(random_state=rs)
            n_cols = self.n_cols.draw_sample(random_state=rs)
            points_image = self._sample_points_for_image(
                image, n_rows, n_cols)
            points.append(points_image)
        
        return points

    def _sample_points_for_image(self, image, n_rows, n_cols):
        height, width = image.shape[0:2]
        
        if n_rows == 1:
            yy = np.array([height // 2], dtype=np.float32)
        else:
            yy = np.linspace(0, height, n_rows)
        
        if n_cols == 1:
            xx = np.array([width // 2], dtype=np.float32)
        else:
            xx = np.linspace(0, width, n_cols)
        
        xx_grid, yy_grid = np.meshgrid(xx, yy)
        return np.stack([xx_grid.flat, yy_grid.flat], axis=-1)

    def __repr__(self):
        return "RegularGridPointsSampler(%s, %s)" % (self.n_rows, self.n_cols)

    def __str__(self):
        return self.__repr__()


class RelativeRegularGridPointsSampler(IPointsSampler):
    """Sample points on relative regular grids."""

    def __init__(self, n_rows_frac, n_cols_frac):
        self.n_rows_frac = iap.handle_continuous_param(
            n_rows_frac, "n_rows_frac", value_range=(0.0+1e-4, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.n_cols_frac = iap.handle_continuous_param(
            n_cols_frac, "n_cols_frac", value_range=(0.0+1e-4, 1.0),
            tuple_to_uniform=True, list_to_choice=True)

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(len(images))
        points = []
        
        for image, rs in zip(images, rss):
            n_rows_frac = self.n_rows_frac.draw_sample(random_state=rs)
            n_cols_frac = self.n_cols_frac.draw_sample(random_state=rs)
            
            height, width = image.shape[0:2]
            n_rows = max(1, int(np.round(n_rows_frac * height)))
            n_cols = max(1, int(np.round(n_cols_frac * width)))
            
            points_image = self._sample_points_for_image(
                image, n_rows, n_cols)
            points.append(points_image)
        
        return points

    def _sample_points_for_image(self, image, n_rows, n_cols):
        height, width = image.shape[0:2]
        
        if n_rows == 1:
            yy = np.array([height // 2], dtype=np.float32)
        else:
            yy = np.linspace(0, height, n_rows)
        
        if n_cols == 1:
            xx = np.array([width // 2], dtype=np.float32)
        else:
            xx = np.linspace(0, width, n_cols)
        
        xx_grid, yy_grid = np.meshgrid(xx, yy)
        return np.stack([xx_grid.flat, yy_grid.flat], axis=-1)

    def __repr__(self):
        return "RelativeRegularGridPointsSampler(%s, %s)" % (
            self.n_rows_frac, self.n_cols_frac)

    def __str__(self):
        return self.__repr__()


class DropoutPointsSampler(IPointsSampler):
    """Apply dropout to points sampled by another sampler."""

    def __init__(self, other_points_sampler, p_drop):
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            "'other_points_sampler', got type %s." % (
                type(other_points_sampler),))
        self.other_points_sampler = other_points_sampler
        self.p_drop = iap.handle_probability_param(p_drop, "p_drop")

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(len(images) + 1)
        points_on_images = self.other_points_sampler.sample_points(
            images, rss[-1])
        
        drop_masks = [self._draw_samples_for_image(points_on_image, rs)
                      for points_on_image, rs
                      in zip(points_on_images, rss[:-1])]
        
        return self._apply_dropout_masks(points_on_images, drop_masks)

    def _draw_samples_for_image(self, points_on_image, random_state):
        drop_samples = self.p_drop.draw_samples((len(points_on_image),),
                                                random_state)
        keep_mask = (drop_samples > 0.5)
        return keep_mask

    @classmethod
    def _apply_dropout_masks(cls, points_on_images, keep_masks):
        points_on_images_dropped = []
        for points_on_image, keep_mask in zip(points_on_images, keep_masks):
            if len(points_on_image) == 0:
                # other sampler didn't provide any points
                poi_dropped = points_on_image
            else:
                if not np.any(keep_mask):
                    # keep at least one point if all were supposed to be dropped
                    idx = (len(points_on_image) - 1) // 2
                    keep_mask = np.copy(keep_mask)
                    keep_mask[idx] = True
                poi_dropped = points_on_image[keep_mask, :]
            points_on_images_dropped.append(poi_dropped)
        return points_on_images_dropped

    def __repr__(self):
        return "DropoutPointsSampler(%s, %s)" % (self.other_points_sampler,
                                                 self.p_drop)

    def __str__(self):
        return self.__repr__()


class UniformPointsSampler(IPointsSampler):
    """Sample points uniformly on images."""

    def __init__(self, n_points):
        self.n_points = iap.handle_discrete_param(
            n_points, "n_points", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(2)
        n_points_imagewise = self._draw_samples(len(images), rss[0])

        n_points_total = np.sum(n_points_imagewise)
        n_components_total = 2 * n_points_total
        coords_relative = rss[1].uniform(0.0, 1.0, n_components_total)
        coords_relative_xy = coords_relative.reshape(n_points_total, 2)

        return self._convert_relative_coords_to_absolute(
            coords_relative_xy, n_points_imagewise, images)

    def _draw_samples(self, n_augmentables, random_state):
        n_points = self.n_points.draw_samples((n_augmentables,),
                                              random_state=random_state)
        n_points_clipped = np.clip(n_points, 1, None)
        return n_points_clipped

    @classmethod
    def _convert_relative_coords_to_absolute(cls, coords_rel_xy,
                                             n_points_imagewise, images):
        coords_absolute = []
        i = 0
        for image, n_points_image in zip(images, n_points_imagewise):
            height, width = image.shape[0:2]
            xx = coords_rel_xy[i:i+n_points_image, 0]
            yy = coords_rel_xy[i:i+n_points_image, 1]

            xx_int = np.clip(np.round(xx * width), 0, width)
            yy_int = np.clip(np.round(yy * height), 0, height)

            coords_absolute.append(np.stack([xx_int, yy_int], axis=-1))
            i += n_points_image
        return coords_absolute

    def __repr__(self):
        return "UniformPointsSampler(%s)" % (self.n_points,)

    def __str__(self):
        return self.__repr__()


class SubsamplingPointsSampler(IPointsSampler):
    """Ensure that the number of sampled points is below a maximum."""

    def __init__(self, other_points_sampler, n_points_max):
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            "'other_points_sampler', got type %s." % (
                type(other_points_sampler),))
        self.other_points_sampler = other_points_sampler
        self.n_points_max = np.clip(n_points_max, -1, None)
        if self.n_points_max == 0:
            ia.warn("Got n_points_max=0 in SubsamplingPointsSampler. "
                    "This will result in no points ever getting returned.")

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(len(images) + 1)
        points_on_images = self.other_points_sampler.sample_points(
            images, rss[-1])
        return [self._subsample(points_on_image, self.n_points_max, rs)
                for points_on_image, rs
                in zip(points_on_images, rss[:-1])]

    @classmethod
    def _subsample(cls, points_on_image, n_points_max, random_state):
        if len(points_on_image) <= n_points_max:
            return points_on_image
        indices = np.arange(len(points_on_image))
        indices_to_keep = random_state.permutation(indices)[0:n_points_max]
        return points_on_image[indices_to_keep]

    def __repr__(self):
        return "SubsamplingPointsSampler(%s, %d)" % (self.other_points_sampler,
                                                     self.n_points_max)

    def __str__(self):
        return self.__repr__()


# ==============================================================================
# TODO LIST
# ==============================================================================

# TODO Add compactness parameter to Superpixels
# TODO Add points subsampler that drops points close to each other first
# TODO Add poisson points sampler
# TODO Add jitter points sampler that moves points around
# for both see https://codegolf.stackexchange.com/questions/50299/draw-an-image-as-a-voronoi-map/50345#50345