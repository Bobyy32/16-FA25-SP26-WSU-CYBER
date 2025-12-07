"""
Augmenters that apply changes to images based on segmentation methods.

List of augmenters:

    * :class:`Superpixels`
    * :class:`Voronoi`
    * :class:`UniformVoronoi`
    * :class:`RegularGridVoronoi`
    * :class:`RelativeRegularGridVoronoi`

"""

from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod

import numpy as np
import skimage.segmentation
import skimage.measure
import six
import six.moves as sm

import imgaug as ia
from . import meta
from .. import random as iarandom
from .. import parameters as iap
from .. import dtypes as iadt
from ..imgaug import _NUMBA_INSTALLED, _numbajit


_SLIC_SUPPORTS_START_LABEL = (
    tuple(map(int, skimage.__version__.split(".")[0:2])) >= (0, 17)
)


def _ensure_image_max_size(image, max_size, interpolation):
    """
    Ensure that images do not exceed a required maximum sidelength.

    This downscales to `max_size` if any side violates that maximum.
    The other side is downscaled too so that the aspect ratio is maintained.

    Parameters
    ----------
    image : ndarray
        Image to potentially downscale.

    max_size : int
        Maximum length of any side of the image.

    interpolation : string or int
        See :func:`~imgaug.imgaug.imresize_single_image`.

    Returns
    -------
    ndarray
        Downscaled image if necessary.

    """
    if max_size is not None:
        size = max(image.shape[0], image.shape[1])
        if size > max_size:
            resize_factor = max_size / size
            new_height = int(image.shape[0] * resize_factor)
            new_width = int(image.shape[1] * resize_factor)
            image = ia.imresize_single_image(
                image, (new_height, new_width), interpolation=interpolation
            )
    return image


class Superpixels(meta.Augmenter):
    """
    Transform images parially/completely to their superpixel representation.

    This implementation uses skimage's version of the SLIC algorithm.

    Parameters
    ----------
    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color.

    n_segments : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Rough target number of how many superpixels to generate.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is exceeded.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Random seed.

    name : None or str, optional
        Name of the augmenter.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.

    deterministic : bool, optional
        Deprecated since 0.4.0.

    """

    def __init__(
        self,
        p_replace=(0.5, 1.0),
        n_segments=(50, 120),
        max_size=128,
        interpolation="linear",
        seed=None,
        name=None,
        random_state="deprecated",
        deterministic="deprecated",
    ):
        super(Superpixels, self).__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True, list_to_choice=True
        )
        self.n_segments = iap.handle_discrete_param(
            n_segments,
            "n_segments",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.max_size = max_size
        self.interpolation = interpolation

    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        iadt.gate_dtypes_strs(
            images,
            allowed="bool uint8 uint16 uint32 uint64 int8 int16 int32 int64",
            disallowed="float16 float32 float64 float128",
            augmenter=self,
        )

        nb_images = len(images)
        rss = random_state.duplicate(1 + nb_images)
        n_segments_samples = self.n_segments.draw_samples(
            (nb_images,), random_state=rss[0]
        )

        n_segments_samples = np.clip(n_segments_samples, 1, None)

        for i, (image, rs) in enumerate(zip(images, rss[1:])):
            if image.size == 0:
                continue

            replace_samples = self.p_replace.draw_samples(
                (n_segments_samples[i],), random_state=rs
            )

            if np.max(replace_samples) == 0:
                continue

            orig_shape = image.shape
            image = _ensure_image_max_size(
                image, self.max_size, self.interpolation
            )

            kwargs = (
                {"start_label": 0} if _SLIC_SUPPORTS_START_LABEL else {}
            )

            segments = skimage.segmentation.slic(
                image, n_segments=n_segments_samples[i], compactness=10, **kwargs
            )

            image_aug = replace_segments_(image, segments, replace_samples > 0.5)

            if orig_shape != image_aug.shape:
                image_aug = ia.imresize_single_image(
                    image_aug, orig_shape[0:2], interpolation=self.interpolation
                )

            batch.images[i] = image_aug
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.p_replace,
            self.n_segments,
            self.max_size,
            self.interpolation,
        ]


def replace_segments_(image, segments, replace_flags):
    """
    Replace segments in images by their average colors in-place.

    Parameters
    ----------
    image : ndarray
        An image of shape ``(H,W,[C])``.

    segments : ndarray
        A ``(H,W)`` integer array containing the same ids for pixels belonging
        to the same segment.

    replace_flags : ndarray or None
        A boolean array denoting whether segments should be replaced.

    Returns
    -------
    ndarray
        The image with replaced pixels.

    """
    assert replace_flags is None or replace_flags.dtype.kind == "b"

    input_shape = image.shape
    if 0 in image.shape:
        return image

    if len(input_shape) == 2:
        image = image[:, :, np.newaxis]

    nb_segments = None
    bad_dtype = image.dtype not in {iadt._UINT8_DTYPE, iadt._INT8_DTYPE}
    if bad_dtype or not _NUMBA_INSTALLED:
        func = _replace_segments_np_
    else:
        max_id = np.max(segments)
        nb_segments = 1 + max_id
        func = _replace_segments_numba_dispatcher_

    result = func(image, segments, replace_flags, nb_segments)

    if len(input_shape) == 2:
        return result[:, :, 0]
    return result


def _replace_segments_np_(image, segments, replace_flags, _nb_segments):
    seg_ids = np.unique(segments)
    if replace_flags is None:
        replace_flags = np.ones((len(seg_ids),), dtype=bool)
    for i, seg_id in enumerate(seg_ids):
        if replace_flags[i % len(replace_flags)]:
            mask = segments == seg_id
            mean_color = np.average(image[mask, :], axis=(0,))
            image[mask] = mean_color
    return image


def _replace_segments_numba_dispatcher_(
    image, segments, replace_flags, nb_segments
):
    if replace_flags is None:
        replace_flags = np.ones((nb_segments,), dtype=bool)
    elif not np.any(replace_flags[:nb_segments]):
        return image

    average_colors = _replace_segments_numba_collect_avg_colors(
        image, segments, replace_flags, nb_segments, image.dtype
    )
    image = _replace_segments_numba_apply_avg_cols_(
        image, segments, replace_flags, average_colors
    )
    return image


@_numbajit(nopython=True, nogil=True, cache=True)
def _replace_segments_numba_collect_avg_colors(
    image, segments, replace_flags, nb_segments, output_dtype
):
    height, width, nb_channels = image.shape
    nb_flags = len(replace_flags)

    average_colors = np.zeros((nb_segments, nb_channels), dtype=np.float64)
    counters = np.zeros((nb_segments,), dtype=np.int32)

    for seg_id in sm.xrange(nb_segments):
        if not replace_flags[seg_id % nb_flags]:
            counters[seg_id] = -1

    for y in sm.xrange(height):
        for x in sm.xrange(width):
            seg_id = segments[y, x]
            count = counters[seg_id]

            if count != -1:
                col = image[y, x, :]
                average_colors[seg_id] += col
                counters[seg_id] += 1

    counters = np.maximum(counters, 1)
    counters = counters.reshape((-1, 1))
    average_colors /= counters

    average_colors = average_colors.astype(output_dtype)
    return average_colors


@_numbajit(nopython=True, nogil=True, cache=True)
def _replace_segments_numba_apply_avg_cols_(
    image, segments, replace_flags, average_colors
):
    height, width = image.shape[0:2]
    nb_flags = len(replace_flags)

    for y in sm.xrange(height):
        for x in sm.xrange(width):
            seg_id = segments[y, x]
            if replace_flags[seg_id % nb_flags]:
                image[y, x, :] = average_colors[seg_id]

    return image


def segment_voronoi(image, cell_coordinates, replace_mask=None):
    """
    Average colors within voronoi cells of an image.

    Parameters
    ----------
    image : ndarray
        The image to convert to a voronoi image.

    cell_coordinates : ndarray
        A ``Nx2`` float array containing the center coordinates of voronoi cells.

    replace_mask : None or ndarray, optional
        Boolean mask denoting for each cell whether its pixels should be replaced.

    Returns
    -------
    ndarray
        Voronoi image.

    """
    input_dims = image.ndim
    if input_dims == 2:
        image = image[..., np.newaxis]

    if len(cell_coordinates) <= 0:
        if input_dims == 2:
            return image[..., 0]
        return image

    height, width = image.shape[0:2]
    ids_of_nearest_cells = _match_pixels_with_voronoi_cells(
        height, width, cell_coordinates
    )
    image_aug = replace_segments_(
        image,
        ids_of_nearest_cells.reshape(image.shape[0:2]),
        replace_mask,
    )

    if input_dims == 2:
        return image_aug[..., 0]
    return image_aug


def _match_pixels_with_voronoi_cells(height, width, cell_coordinates):
    from scipy.spatial import cKDTree as KDTree

    tree = KDTree(cell_coordinates)
    pixel_coords = _generate_pixel_coords(height, width)
    pixel_coords_subpixel = pixel_coords.astype(np.float32) + 0.5
    ids_of_nearest_cells = tree.query(pixel_coords_subpixel)[1]
    return ids_of_nearest_cells


def _generate_pixel_coords(height, width):
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    return np.c_[xx.ravel(), yy.ravel()]


class Voronoi(meta.Augmenter):
    """
    Average colors of an image within Voronoi cells.

    Parameters
    ----------
    points_sampler : IPointsSampler
        A points sampler which will be queried per image to generate the
        coordinates of the centers of voronoi cells.

    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is exceeded.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Random seed.

    name : None or str, optional
        Name of the augmenter.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.

    deterministic : bool, optional
        Deprecated since 0.4.0.

    """

    def __init__(
        self,
        points_sampler,
        p_replace=1.0,
        max_size=128,
        interpolation="linear",
        seed=None,
        name=None,
        random_state="deprecated",
        deterministic="deprecated",
    ):
        super(Voronoi, self).__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        assert isinstance(points_sampler, IPointsSampler), (
            "Expected 'points_sampler' to be an instance of IPointsSampler, "
            "got %s." % (type(points_sampler),)
        )
        self.points_sampler = points_sampler

        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True, list_to_choice=True
        )

        self.max_size = max_size
        self.interpolation = interpolation

    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        iadt.allow_only_uint8(images, augmenter=self)

        rss = random_state.duplicate(len(images))
        for i, (image, rs) in enumerate(zip(images, rss)):
            batch.images[i] = self._augment_single_image(image, rs)
        return batch

    def _augment_single_image(self, image, random_state):
        rss = random_state.duplicate(2)
        orig_shape = image.shape
        image = _ensure_image_max_size(image, self.max_size, self.interpolation)

        cell_coordinates = self.points_sampler.sample_points([image], rss[0])[0]
        p_replace = self.p_replace.draw_samples(
            (len(cell_coordinates),), rss[1]
        )
        replace_mask = p_replace > 0.5

        image_aug = segment_voronoi(image, cell_coordinates, replace_mask)

        if orig_shape != image_aug.shape:
            image_aug = ia.imresize_single_image(
                image_aug, orig_shape[0:2], interpolation=self.interpolation
            )

        return image_aug

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.points_sampler,
            self.p_replace,
            self.max_size,
            self.interpolation,
        ]


class UniformVoronoi(Voronoi):
    """
    Uniformly sample Voronoi cells on images and average colors within them.

    Parameters
    ----------
    n_points : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of points to sample on each image.

    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is exceeded.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Random seed.

    name : None or str, optional
        Name of the augmenter.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.

    deterministic : bool, optional
        Deprecated since 0.4.0.

    """

    def __init__(
        self,
        n_points=(50, 500),
        p_replace=(0.5, 1.0),
        max_size=128,
        interpolation="linear",
        seed=None,
        name=None,
        random_state="deprecated",
        deterministic="deprecated",
    ):
        super(UniformVoronoi, self).__init__(
            points_sampler=UniformPointsSampler(n_points),
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class RegularGridVoronoi(Voronoi):
    """
    Sample Voronoi cells from regular grids and color-average them.

    Parameters
    ----------
    n_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of rows of coordinates to place on each image.

    n_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of columns of coordinates to place on each image.

    p_drop_points : number or tuple of number or imgaug.parameters.StochasticParameter, optional
        The probability that a coordinate will be removed.

    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is exceeded.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Random seed.

    name : None or str, optional
        Name of the augmenter.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.

    deterministic : bool, optional
        Deprecated since 0.4.0.

    """

    def __init__(
        self,
        n_rows=(10, 30),
        n_cols=(10, 30),
        p_drop_points=(0.0, 0.5),
        p_replace=(0.5, 1.0),
        max_size=128,
        interpolation="linear",
        seed=None,
        name=None,
        random_state="deprecated",
        deterministic="deprecated",
    ):
        super(RegularGridVoronoi, self).__init__(
            points_sampler=DropoutPointsSampler(
                RegularGridPointsSampler(n_rows, n_cols), p_drop_points
            ),
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class RelativeRegularGridVoronoi(Voronoi):
    """
    Sample Voronoi cells from image-dependent grids and color-average them.

    Parameters
    ----------
    n_rows_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the y-axis.

    n_cols_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the x-axis.

    p_drop_points : number or tuple of number or imgaug.parameters.StochasticParameter, optional
        The probability that a coordinate will be removed.

    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is exceeded.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Random seed.

    name : None or str, optional
        Name of the augmenter.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.

    deterministic : bool, optional
        Deprecated since 0.4.0.

    """

    def __init__(
        self,
        n_rows_frac=(0.05, 0.15),
        n_cols_frac=(0.05, 0.15),
        p_drop_points=(0.0, 0.5),
        p_replace=(0.5, 1.0),
        max_size=None,
        interpolation="linear",
        seed=None,
        name=None,
        random_state="deprecated",
        deterministic="deprecated",
    ):
        super(RelativeRegularGridVoronoi, self).__init__(
            points_sampler=DropoutPointsSampler(
                RelativeRegularGridPointsSampler(n_rows_frac, n_cols_frac),
                p_drop_points,
            ),
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@six.add_metaclass(ABCMeta)
class IPointsSampler(object):
    """
    Interface for all point samplers.

    Point samplers return coordinate arrays of shape ``Nx2``.

    """

    @abstractmethod
    def sample_points(self, images, random_state):
        """
        Generate coordinates of points on images.

        Parameters
        ----------
        images : ndarray or list of ndarray
            One or more images for which to generate points.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState
            A random state to use for any probabilistic function required
            during the point sampling.

        Returns
        -------
        ndarray
            An ``(N,2)`` ``float32`` array containing ``(x,y)`` subpixel coordinates.

        """


def _verify_sample_points_images(images):
    assert len(images) > 0, "Expected at least one image, got zero."
    if isinstance(images, list):
        assert all([ia.is_np_array(image) for image in images]), (
            "Expected list of numpy arrays, got list of types %s."
            % (", ".join([str(type(image)) for image in images]),)
        )
        assert all([image.ndim == 3 for image in images]), (
            "Expected each image to have three dimensions, "
            "got dimensions %s." % (", ".join([str(image.ndim) for image in images]),)
        )
    else:
        assert ia.is_np_array(images), (
            "Expected either a list of numpy arrays or a single numpy "
            "array of shape NxHxWxC. Got type %s." % (type(images),)
        )
        assert images.ndim == 4, (
            "Expected a four-dimensional array of shape NxHxWxC. "
            "Got shape %d dimensions (shape: %s)." % (images.ndim, images.shape)
        )


class RegularGridPointsSampler(IPointsSampler):
    """
    Sampler that generates a regular grid of coordinates on an image.

    Parameters
    ----------
    n_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of rows of coordinates to place on each image.

    n_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of columns of coordinates to place on each image.

    """

    def __init__(self, n_rows, n_cols):
        self.n_rows = iap.handle_discrete_param(
            n_rows,
            "n_rows",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.n_cols = iap.handle_discrete_param(
            n_cols,
            "n_cols",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        n_rows_lst, n_cols_lst = self._draw_samples(images, random_state)
        return self._generate_point_grids(images, n_rows_lst, n_cols_lst)

    def _draw_samples(self, images, random_state):
        rss = random_state.duplicate(2)
        n_rows_lst = self.n_rows.draw_samples(len(images), random_state=rss[0])
        n_cols_lst = self.n_cols.draw_samples(len(images), random_state=rss[1])
        return self._clip_rows_and_cols(n_rows_lst, n_cols_lst, images)

    @classmethod
    def _clip_rows_and_cols(cls, n_rows_lst, n_cols_lst, images):
        heights = np.int32([image.shape[0] for image in images])
        widths = np.int32([image.shape[1] for image in images])
        n_rows_lst = np.clip(n_rows_lst, None, heights)
        n_cols_lst = np.clip(n_cols_lst, None, widths)
        n_rows_lst = np.clip(n_rows_lst, 1, None)
        n_cols_lst = np.clip(n_cols_lst, 1, None)
        return n_rows_lst, n_cols_lst

    @classmethod
    def _generate_point_grids(cls, images, n_rows_lst, n_cols_lst):
        grids = []
        for image, n_rows_i, n_cols_i in zip(images, n_rows_lst, n_cols_lst):
            grids.append(cls._generate_point_grid(image, n_rows_i, n_cols_i))
        return grids

    @classmethod
    def _generate_point_grid(cls, image, n_rows, n_cols):
        height, width = image.shape[0:2]

        y_spacing = height / n_rows
        y_start = 0.0 + y_spacing / 2
        y_end = height - y_spacing / 2
        if y_start - 1e-4 <= y_end <= y_start + 1e-4:
            yy = np.float32([y_start])
        else:
            yy = np.linspace(y_start, y_end, num=n_rows)

        x_spacing = width / n_cols
        x_start = 0.0 + x_spacing / 2
        x_end = width - x_spacing / 2
        if x_start - 1e-4 <= x_end <= x_start + 1e-4:
            xx = np.float32([x_start])
        else:
            xx = np.linspace(x_start, x_end, num=n_cols)

        xx, yy = np.meshgrid(xx, yy)
        grid = np.vstack([xx.ravel(), yy.ravel()]).T
        return grid

    def __repr__(self):
        return "RegularGridPointsSampler(%s, %s)" % (self.n_rows, self.n_cols)

    def __str__(self):
        return self.__repr__()


class RelativeRegularGridPointsSampler(IPointsSampler):
    """
    Regular grid coordinate sampler; places more points on larger images.

    Parameters
    ----------
    n_rows_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the y-axis.

    n_cols_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the x-axis.

    """

    def __init__(self, n_rows_frac, n_cols_frac):
        eps = 1e-4
        self.n_rows_frac = iap.handle_continuous_param(
            n_rows_frac,
            "n_rows_frac",
            value_range=(0.0 + eps, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.n_cols_frac = iap.handle_continuous_param(
            n_cols_frac,
            "n_cols_frac",
            value_range=(0.0 + eps, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
        )

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        n_rows, n_cols = self._draw_samples(images, random_state)
        return RegularGridPointsSampler._generate_point_grids(images, n_rows, n_cols)

    def _draw_samples(self, images, random_state):
        n_augmentables = len(images)
        rss = random_state.duplicate(2)
        n_rows_frac = self.n_rows_frac.draw_samples(n_augmentables, random_state=rss[0])
        n_cols_frac = self.n_cols_frac.draw_samples(n_augmentables, random_state=rss[1])
        heights = np.int32([image.shape[0] for image in images])
        widths = np.int32([image.shape[1] for image in images])

        n_rows = np.round(n_rows_frac * heights)
        n_cols = np.round(n_cols_frac * widths)
        n_rows, n_cols = RegularGridPointsSampler._clip_rows_and_cols(
            n_rows, n_cols, images
        )

        return n_rows.astype(np.int32), n_cols.astype(np.int32)

    def __repr__(self):
        return "RelativeRegularGridPointsSampler(%s, %s)" % (
            self.n_rows_frac,
            self.n_cols_frac,
        )

    def __str__(self):
        return self.__repr__()


class DropoutPointsSampler(IPointsSampler):
    """
    Remove a defined fraction of sampled points.

    Parameters
    ----------
    other_points_sampler : IPointsSampler
        Another point sampler that is queried to generate a list of points.

    p_drop : number or tuple of number or imgaug.parameters.StochasticParameter
        The probability that a coordinate will be removed.

    """

    def __init__(self, other_points_sampler, p_drop):
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            "'other_points_sampler', got type %s." % (type(other_points_sampler),)
        )
        self.other_points_sampler = other_points_sampler
        self.p_drop = self._convert_p_drop_to_inverted_mask_param(p_drop)

    @classmethod
    def _convert_p_drop_to_inverted_mask_param(cls, p_drop):
        if ia.is_single_number(p_drop):
            p_drop = iap.Binomial(1 - p_drop)
        elif ia.is_iterable(p_drop):
            assert len(p_drop) == 2, (
                "Expected 'p_drop' given as an iterable to contain exactly "
                "2 values, got %d." % (len(p_drop),)
            )
            assert p_drop[0] < p_drop[1], (
                "Expected 'p_drop' given as iterable to contain exactly 2 "
                "values (a, b) with a < b. Got %.4f and %.4f." % (p_drop[0], p_drop[1])
            )
            assert 0 <= p_drop[0] <= 1.0 and 0 <= p_drop[1] <= 1.0, (
                "Expected 'p_drop' given as iterable to only contain values "
                "in the interval [0.0, 1.0], got %.4f and %.4f." % (p_drop[0], p_drop[1])
            )
            p_drop = iap.Binomial(iap.Uniform(1 - p_drop[1], 1 - p_drop[0]))
        elif isinstance(p_drop, iap.StochasticParameter):
            pass
        else:
            raise Exception(
                "Expected p_drop to be float or int or StochasticParameter, "
                "got %s." % (type(p_drop),)
            )
        return p_drop

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(2)
        points_on_images = self.other_points_sampler.sample_points(images, rss[0])
        drop_masks = self._draw_samples(points_on_images, rss[1])
        return self._apply_dropout_masks(points_on_images, drop_masks)

    def _draw_samples(self, points_on_images, random_state):
        rss = random_state.duplicate(len(points_on_images))
        drop_masks = [
            self._draw_samples_for_image(points_on_image, rs)
            for points_on_image, rs in zip(points_on_images, rss)
        ]
        return drop_masks

    def _draw_samples_for_image(self, points_on_image, random_state):
        drop_samples = self.p_drop.draw_samples(
            (len(points_on_image),), random_state
        )
        keep_mask = drop_samples > 0.5
        return keep_mask

    @classmethod
    def _apply_dropout_masks(cls, points_on_images, keep_masks):
        points_on_images_dropped = []
        for points_on_image, keep_mask in zip(points_on_images, keep_masks):
            if len(points_on_image) == 0:
                poi_dropped = points_on_image
            else:
                if not np.any(keep_mask):
                    idx = (len(points_on_image) - 1) // 2
                    keep_mask = np.copy(keep_mask)
                    keep_mask[idx] = True
                poi_dropped = points_on_image[keep_mask, :]
            points_on_images_dropped.append(poi_dropped)
        return points_on_images_dropped

    def __repr__(self):
        return "DropoutPointsSampler(%s, %s)" % (self.other_points_sampler, self.p_drop)

    def __str__(self):
        return self.__repr__()


class UniformPointsSampler(IPointsSampler):
    """
    Sample points uniformly on images.

    Parameters
    ----------
    n_points : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of points to sample on each image.

    """

    def __init__(self, n_points):
        self.n_points = iap.handle_discrete_param(
            n_points,
            "n_points",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

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
            coords_relative_xy, n_points_imagewise, images
        )

    def _draw_samples(self, n_augmentables, random_state):
        n_points = self.n_points.draw_samples((n_augmentables,), random_state=random_state)
        n_points_clipped = np.clip(n_points, 1, None)
        return n_points_clipped

    @classmethod
    def _convert_relative_coords_to_absolute(cls, coords_rel_xy, n_points_imagewise, images):
        coords_absolute = []
        i = 0
        for image, n_points_image in zip(images, n_points_imagewise):
            height, width = image.shape[0:2]
            xx = coords_rel_xy[i : i + n_points_image, 0]
            yy = coords_rel_xy[i : i + n_points_image, 1]

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
    """
    Ensure that the number of sampled points is below a maximum.

    Parameters
    ----------
    other_points_sampler : IPointsSampler
        Another point sampler that is queried to generate a list of points.

    n_points_max : int
        Maximum number of allowed points.

    """

    def __init__(self, other_points_sampler, n_points_max):
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            "'other_points_sampler', got type %s." % (type(other_points_sampler),)
        )
        self.other_points_sampler = other_points_sampler
        self.n_points_max = np.clip(n_points_max, -1, None)
        if self.n_points_max == 0:
            ia.warn(
                "Got n_points_max=0 in SubsamplingPointsSampler. "
                "This will result in no points ever getting returned."
            )

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(len(images) + 1)
        points_on_images = self.other_points_sampler.sample_points(images, rss[-1])
        return [
            self._subsample(points_on_image, self.n_points_max, rs)
            for points_on_image, rs in zip(points_on_images, rss[:-1])
        ]

    @classmethod
    def _subsample(cls, points_on_image, n_points_max, random_state):
        if len(points_on_image) <= n_points_max:
            return points_on_image
        indices = np.arange(len(points_on_image))
        indices_to_keep = random_state.permutation(indices)[0:n_points_max]
        return points_on_image[indices_to_keep]

    def __repr__(self):
        return "SubsamplingPointsSampler(%s, %d)" % (
            self.other_points_sampler,
            self.n_points_max,
        )

    def __str__(self):
        return self.__repr__()