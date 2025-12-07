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
# use skimage.segmentation instead `from skimage import segmentation` here,
# because otherwise unittest seems to mix up imgaug.augmenters.segmentation
# with skimage.segmentation for whatever reason
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
    tuple(map(int, skimage.__version__.split(".")[0:2]))
    >= (0, 17)
)  # Added in 0.5.0.


# TODO merge this into imresize?
def _ensure_image_max_size(img, max_size, interp):
    """Ensure that images do not exceed a required maximum sidelength.

    This downscales to `max_size` if any side violates that maximum.
    The other side is downscaled too so that the aspect ratio is maintained.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.imresize_single_image`.

    Parameters
    ----------
    img : ndarray
        Image to potentially downscale.

    max_size : int
        Maximum length of any side of the image.

    interp : string or int
        See :func:`~imgaug.imgaug.imresize_single_image`.

    """
    if max_size is not None:
        size = max(img.shape[0], img.shape[1])
        if size > max_size:
            factor = max_size / size
            new_h = int(img.shape[0] * factor)
            new_w = int(img.shape[1] * factor)
            img = ia.imresize_single_image(
                img,
                (new_h, new_w),
                interpolation=interp)
    return img


# TODO add compactness parameter
class Superpixels(meta.Augmenter):
    """Transform images parially/completely to their superpixel representation.

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
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that ``number`` will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    n_segments : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Rough target number of how many superpixels to generate (the algorithm
        may deviate from this number). Lower value will lead to coarser
        superpixels. Higher values are computationally more intensive and
        will hence lead to a slowdown.

            * If a single ``int``, then that value will always be used as the
              number of segments.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interp : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug.imgaug.imresize_single_image`.

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
    >>> aug = iaa.Superpixels(p_replace=1.0, n_segments=64)

    Replace all pixels in each segment with their average value.

    >>> aug = iaa.Superpixels(p_replace=0.5, n_segments=64)

    Replace pixels in half of all segments with their average values.

    >>> aug = iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128))

    Replace pixels in ``25%`` to ``100%`` of all segments with their
    average values. Use ``16`` to ``128`` segments per image.

    >>> aug = iaa.Superpixels(
    >>>     p_replace=iap.Beta(0.5, 0.5),
    >>>     n_segments=iap.Choice([64, 128])
    >>> )

    Replace pixels in a percentage ``p`` of all segments with their
    average values, where ``p`` is sampled from ``Beta(0.5, 0.5)``.
    Use either ``64`` or ``128`` segments per image.

    """

    def __init__(self, p_replace=0, n_segments=(50, 120), max_size=128,
                 interp="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Superpixels, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True,
            list_to_choice=True)
        self.n_segments = iap.handle_discrete_param(
            n_segments, "n_segments", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.max_size = max_size
        self.interp = interp

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        imgs = batch.images

        iadt.gate_dtypes(
            imgs,
            allowed=["bool",
                     "uint8", "uint16", "uint32",
                     "int8", "int16", "int32"],
            disallowed=["uint64", "uint128", "uint256",
                         "int64", "int128", "int256",
                         "float16", "float32", "float64", "float96",
                         "float128", "float256"],
            augmenter=self)

        batch.images = list(imgs)
        samples = self._draw_samples(batch, random_state)
        for i, (img, s) in enumerate(zip(batch.images, samples)):
            batch.images[i] = self._transform_single_img(img, s)
        return batch

    def _draw_samples(self, batch, rs):
        n_imgs = batch.nb_rows
        rss = rs.duplicate(2 * n_imgs)

        # (1) get nb of segments per img
        n_segs_per_img = self.n_segments.draw_samples(
            (n_imgs,), random_state=rss[0])

        # (2) for images with replaced pixels, get probability per img (not
        # per superpixel) -- these are the same values that are returned
        # per image in the `samples` list
        p_replace_samples = self.p_replace.draw_samples(
            (n_imgs,), random_state=rss[1])

        # (3) per image that has at least one superpixel which's pixels
        # are replaced within the superpixel, derive the probability
        # per superpixel
        rss_imgs = rss[2:]
        samples = []
        for i, (n_segs, p_replace, rss_img) \
                in enumerate(zip(n_segs_per_img, p_replace_samples, rss_imgs)):
            if p_replace >= 0.5:
                # image with replaced pixels
                mask = self.p_replace.draw_samples(
                    (n_segs,), random_state=rss_img)
                # TODO this results in averaging over uint8 if the image
                #      is uint8 and leads to inaccuracies
                # TODO make more flexible
                mask = mask >= 0.5
            else:
                # image without replaced pixels
                mask = np.zeros((n_segs,), dtype=bool)

            sample = _Superpixels_SampleData(
                n_segments=n_segs,
                replace_mask=mask)
            samples.append(sample)

        return samples

    def _transform_single_img(self, img, s):
        h, w, c = iadt.restore_dtypes_(img.shape, {2: 1})[0:3]

        img_sp = img
        img_sp = _ensure_image_max_size(img_sp, self.max_size, self.interp)
        if img_sp.ndim != img.ndim:
            h_sp, w_sp, c_sp = iadt.restore_dtypes_(img_sp.shape, {2: 1})[0:3]
            h_sp, w_sp, c_sp = h_sp, w_sp, 1 if c_sp is None else c_sp

            assert c == c_sp, (
                "Got an unexpected number of channels "
                "after downsampling a image from shape %s to %s in "
                "Superpixels augmenter. Input image's number of channels "
                "was %d. Expected number of channels after downsampling "
                "was also %d, but got %d." % (
                    img.shape, img_sp.shape, c, c, c_sp))

        segs = skimage.segmentation.slic(
            img_sp, n_segments=s.n_segments, compactness=10,
            **_SLIC_DICT_SUPPORTS_START_LABEL)

        n_segs = len(np.unique(segs))
        assert n_segs <= len(s.replace_mask), (
            "Got %d unique superpixels from slic(), but only %d "
            "values in replace_mask. Requested number of superpixels "
            "was %d." % (n_segs, len(s.replace_mask), s.n_segments))

        if np.any(s.replace_mask):
            img_sp_c3 = iadt.restore_dtypes_(
                img_sp, np.uint8, clip=True)[0]
            img_sp_c3 = iadt.ensure_default_dtype(img_sp_c3)
            if not iadt.is_iterable_of_dtype(
                    img_sp_c3, "uint8", check_all_elements=False):
                img_sp_c3 = img_sp_c3.astype(np.uint8)

            img_sp_replaced = _replace_segs(
                img_sp_c3, segs, s.replace_mask)
            if img_sp_replaced.ndim == 2 and c is not None and c > 1:
                img_sp_replaced = np.tile(
                    img_sp_replaced[..., np.newaxis], (1, 1, c))
            if img.shape[0:2] != img_sp_replaced.shape[0:2]:
                img_sp_replaced = ia.imresize_single_image(
                    img_sp_replaced, img.shape[0:2],
                    interpolation=self.interp)

            img = img_sp_replaced
        elif self.max_size is not None:
            img = ia.imresize_single_image(
                img, img.shape[0:2], interpolation=self.interp)

        return img

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p_replace, self.n_segments, self.max_size,
                self.interp]


class _Superpixels_SampleData(object):
    def __init__(self, n_segments, replace_mask):
        self.n_segments = n_segments
        self.replace_mask = replace_mask


_SLIC_DICT_SUPPORTS_START_LABEL = (
    {"start_label": 0} if _SLIC_SUPPORTS_START_LABEL else {})


def _replace_segs(img, segs, replace_mask):
    if _NUMBA_INSTALLED:
        return _replace_segs_numba(img, segs, replace_mask)
    return _replace_segs_numpy(img, segs, replace_mask)


def _replace_segs_numpy(img, segs, replace_mask):
    img_sp = img.copy()
    n_c = 1 if img_sp.ndim == 2 else img_sp.shape[-1]
    for c in sm.xrange(n_c):
        # segments in replace_mask are not necessarily the
        # same as in markers, i.e. replace_mask[0] does not have to
        # affect segment with id 0 in markers, it affects instead the
        # segment with value 0 at LA.markers[0, 0],
        # which can also be any other value
        seg_ids = np.unique(segs).astype(np.int32)
        seg_ids_pos = np.arange(len(seg_ids)).astype(np.int32)
        seg_ids_mask_map = {seg_ids[i]: replace_mask[i]
                             for i in seg_ids_pos}

        # TODO this is very inefficient to run per channel
        # remap markers to 0, 1, 2, ...
        segs_mask_map = np.zeros((segs.max()+1,), dtype=bool)
        for idx, should_replace in seg_ids_mask_map.items():
            segs_mask_map[idx] = should_replace
        segs_mask = segs_mask_map[segs]

        if n_c == 1:
            img_sp[segs_mask] = 0
        else:
            img_sp[segs_mask, c] = 0

    regions = skimage.measure.regionprops(segs, intensity_image=img)

    for ridx, region in enumerate(regions):
        if replace_mask[ridx]:
            # TODO for a proper per-channel average, we would have to compute
            # `mean_intensity` here for every channel instead of once for all
            # channels
            mean_intensity = region.mean_intensity
            if img_sp.ndim == 2:
                img_sp[segs == ridx] = mean_intensity
            elif img_sp.ndim == 3:
                img_sp[segs == ridx] = mean_intensity

    return img_sp


@_numbajit("uint8[:,:,::1](uint8[:,:,::1],int32[:,:],bool_[:])",
           nopython=True, nogil=True, cache=True)
def _replace_segs_numba(img, segs, replace_mask):
    n_c = img.shape[-1]
    img_sp = np.copy(img)
    seg_ids = np.unique(segs)
    for seg_id, seg_mask_i in zip(seg_ids, replace_mask):
        if not seg_mask_i:
            continue
        mask = (segs == seg_id)
        n_px_in_seg = np.sum(mask)
        for c in range(n_c):
            c_sum = 0
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    if mask[y, x]:
                        c_sum += img[y, x, c]
            c_avg = int(np.round(c_sum / max(n_px_in_seg, 1)))
            for y in range(img_sp.shape[0]):
                for x in range(img_sp.shape[1]):
                    if mask[y, x]:
                        img_sp[y, x, c] = c_avg
    return img_sp


# for whatever reason, having Voronoi in this module can crash the python
# interpreter
from . _artistic import Voronoi  # noqa
Voronoi.__module__ = "imgaug.augmenters.segmentation"


# TODO maybe rename this to PointsVoronoi or PointsSampler?
# TODO add tests for this
class IPointsSampler(six.with_metaclass(ABCMeta, object)):
    """Interface for all point samplers.

    Point samplers return a list of points that can be used to place
    voronoi cells on images.

    All point samplers must be derived from this interface.

    """

    def __init__(self):
        pass

    @abstractmethod
    def sample_points(self, imgs, rs):
        """Generate coordinates of points on images.

        Parameters
        ----------
        imgs : list of ndarray
            Images on which to generate points.

        rs : imgaug.random.RNG
            Random state to use.

        Returns
        -------
        list of ndarray
            List of ``(N,2)`` ``float32`` arrays containing point coordinates
            for the ``N`` points placed on each image, given as ``(x,y)``
            coordinates.

        """


def _verify_sample_points_images(imgs):
    assert isinstance(imgs, list), (
        "Expected 'imgs' to be a list, got type %s." % (type(imgs),))
    assert len(imgs) > 0, (
        "Cannot sample points on zero images.")
    for img in imgs:
        # we dont use is_np_array() here, because that would allow 2D
        # images and the below ensures 3D images
        assert hasattr(img, "shape") and hasattr(img, "dtype"), (
            "Expected each image to have 'shape' and 'dtype' attributes, "
            "got type %s." % (type(img),))
        assert len(img.shape) == 3, (
            "Expected 3-dimensional images, got shape %s." % (img.shape,))
        assert img.shape[0] > 0 and img.shape[1] > 0, (
            "Expected height and width of each image to be greater than "
            "zero, got shape %s." % (img.shape,))


# TODO allow to change placement method (grid, poisson disk sampling, ...)
class RegularGridPointsSampler(IPointsSampler):
    """Sample points on a regular grid over images.

    This point sampler generates a regular grid with `H` rows and `W` cols
    over each image. Then, uniformly places a point on each cell of the grid.

    Parameters
    ----------
    n_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Number of rows of coordinates to place on each image, i.e. the number
        of coordinates on the y-axis. Note that for each image, the sampled
        value is clipped to the minimum of the image's height and the sampled
        value.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Number of columns of coordinates to place on each image, i.e. the
        number of coordinates on the x-axis. Note that for each image, the
        sampled value is clipped to the minimum of the image's width and the
        sampled value.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.RegularGridPointsSampler(10, 20)

    Create a point sampler that generates a ``10x20`` point grid on each
    image. Then, a point is placed on each cell of the grid. Each point is
    placed uniformly within the cell.

    >>> sampler = iaa.RegularGridPointsSampler(10, iap.DeterministicList([10, 12, 14, 16, 18, 20]))

    Create a point sampler that generates a ``10xW`` point grid on each
    image, with ``W`` being sampled from the list ``[10, 12, 14, 16, 18, 20]``
    (a different ``W`` value can be sampled per image).

    """

    def __init__(self, n_rows, n_cols):
        self.n_rows = iap.handle_discrete_param(
            n_rows, "n_rows", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.n_cols = iap.handle_discrete_param(
            n_cols, "n_cols", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)

    def sample_points(self, imgs, rs):
        rs = iarandom.RNG.create_if_not_rng_(rs)
        _verify_sample_points_images(imgs)

        rss = rs.duplicate(len(imgs))

        n_rows_imgs = self.n_rows.draw_samples(
            (len(imgs),), random_state=rs)
        n_cols_imgs = self.n_cols.draw_samples(
            (len(imgs),), random_state=rs)

        points_imgs = []
        for img, n_rows_img, n_cols_img, rs_img \
                in zip(imgs, n_rows_imgs, n_cols_imgs, rss):
            h, w = img.shape[0:2]

            n_rows_img = min(n_rows_img, h)
            n_cols_img = min(n_cols_img, w)

            points_img = self._sample_points_for_img(
                h, w, n_rows_img, n_cols_img, rs_img)
            points_imgs.append(points_img)

        return points_imgs

    @classmethod
    def _sample_points_for_img(cls, h, w, n_rows, n_cols,
                                     rs):
        n_cells = n_rows * n_cols
        if n_cells == 0:
            return np.zeros((0, 2), dtype=np.float32)

        cell_h = h / n_rows
        cell_w = w / n_cols

        # we sample coordinates in range [0.0, 1.0] and later multiply
        # by the cell width/height to get the distance from the cell
        # origin on each axis.
        # We could also sample directly in the range [0, cell_width],
        # but that would likely not be faster.
        coords_rel = rs.uniform(0.0, 1.0, size=(2 * n_cells,))
        coords_rel = coords_rel.reshape((n_cells, 2))

        # for cell 0 at row 0, col 0: cell_y_start=0, cell_x_start=0
        # for cell 1 at row 0, col 1: cell_y_start=0, cell_x_start=cell_w
        # for cell 2 at row 0, col 2: cell_y_start=0, cell_x_start=2*cell_w
        # for cell 3 at row 1, col 0: cell_y_start=cell_h, cell_x_start=0
        cell_y_start = np.tile(
            np.linspace(0, h - cell_h, n_rows).reshape(-1, 1),
            (1, n_cols)
        ).flatten()
        cell_x_start = np.tile(
            np.linspace(0, w - cell_w, n_cols).reshape(1, -1),
            (n_rows, 1)
        ).flatten()

        points_y = cell_y_start + cell_h * coords_rel[:, 0]
        points_x = cell_x_start + cell_w * coords_rel[:, 1]

        points_img = np.stack([points_x, points_y], axis=-1)
        points_img = points_img.astype(np.float32)

        return points_img

    def __repr__(self):
        return "RegularGridPointsSampler(%s, %s)" % (self.n_rows, self.n_cols)

    def __str__(self):
        return self.__repr__()


class RelativeRegularGridPointsSampler(IPointsSampler):
    """Sample points on a regular grid over images.

    This point sampler generates a regular grid with ``H*n_rows_frac`` rows
    and ``W*n_cols_frac`` cols over each image, where ``H`` is the image
    height and ``W`` is the image width. Then, a point is placed on each cell
    of the grid.

    Parameters
    ----------
    n_rows_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of rows of coordinates to place on each image, i.e.
        the number of coordinates on the y-axis.
        For ``S`` rows and image height ``H``, the number of actually placed
        rows is ``S * H``. Note that for each image, the number of placed
        rows is clipped to ``H``.

            * If a ``number``, then that ``number`` will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the continuous
              interval ``[a, b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of columns of coordinates to place on each image,
        i.e. the number of coordinates on the x-axis. Analogous to
        `n_rows_frac`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.RelativeRegularGridPointsSampler(0.1, 0.05)

    Create a point sampler that generates a regular grid with ``0.1*H`` rows
    and ``0.05*W`` columns on each image (with ``H`` being the image height
    and ``W`` the width). Then, a point is uniformly placed on each cell of
    the grid.

    >>> sampler = iaa.RelativeRegularGridPointsSampler(
    >>>     iap.Uniform(0.05, 0.15),
    >>>     0.05
    >>> )

    Create a point sampler that generates a regular grid with ``R*H`` rows
    and ``0.05*W`` cols on each image (with ``H`` being the image height,
    ``W`` the width and ``R`` being a random value between ``0.05`` and
    ``0.15``).

    """

    def __init__(self, n_rows_frac, n_cols_frac):
        self.n_rows_frac = iap.handle_continuous_param(
            n_rows_frac, "n_rows_frac", value_range=(0.0+1e-4, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.n_cols_frac = iap.handle_continuous_param(
            n_cols_frac, "n_cols_frac", value_range=(0.0+1e-4, 1.0),
            tuple_to_uniform=True, list_to_choice=True)

    def sample_points(self, imgs, rs):
        rs = iarandom.RNG.create_if_not_rng_(rs)
        _verify_sample_points_images(imgs)

        rss = rs.duplicate(len(imgs))

        n_rows_fracs = self.n_rows_frac.draw_samples(
            (len(imgs),), random_state=rs)
        n_cols_fracs = self.n_cols_frac.draw_samples(
            (len(imgs),), random_state=rs)

        points_imgs = []
        for img, n_rows_frac, n_cols_frac, rs_img \
                in zip(imgs, n_rows_fracs, n_cols_fracs, rss):
            h, w = img.shape[0:2]

            n_rows = int(np.round(n_rows_frac * h))
            n_rows = max(n_rows, 1)
            n_rows = min(n_rows, h)

            n_cols = int(np.round(n_cols_frac * w))
            n_cols = max(n_cols, 1)
            n_cols = min(n_cols, w)

            points_img = RegularGridPointsSampler._sample_points_for_img(
                h, w, n_rows, n_cols, rs_img)
            points_imgs.append(points_img)

        return points_imgs

    def __repr__(self):
        return "RelativeRegularGridPointsSampler(%s, %s)" % (
            self.n_rows_frac, self.n_cols_frac)

    def __str__(self):
        return self.__repr__()


class DropoutPointsSampler(IPointsSampler):
    """Remove random points from collections of points.

    This point sampler takes other point samplers and removes
    ``N*p_drop`` of the sampled points, where ``N`` is the number of
    points.

    Parameters
    ----------
    other_sampler : IPointsSampler
        Another point sampler that is queried to generate a list of points.
        The dropout operation will be applied to that list.

    p_drop : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The probability that a coordinate will be removed from the list
        of all sampled coordinates. A value of ``1.0`` would mean that (on
        average) all coordinates will be dropped, ``0.0`` that none will
        be dropped. Note that at least one coordinate will always be left
        over and never be dropped.

            * If a ``float``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a random value will be sampled
              from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image. It is *recommended* to
              use e.g. ``Beta(0.5, 0.5)`` or ``TruncatedNormal(0.5, 0.25, 0, 1)``.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.DropoutPointsSampler(
    >>>     iaa.RelativeRegularGridPointsSampler(0.1, 0.2),
    >>>     0.2
    >>> )

    Create a point sampler that first generates ``y*H`` points on the y-axis
    (with ``y`` being ``0.1`` and ``H`` the image height) and analogously
    ``x*W`` on the x axis. Then, the sampler drops ``20%`` of these points.
    At least one point will be left over.

    """

    def __init__(self, other_sampler, p_drop):
        assert isinstance(other_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            "'other_sampler', got type %s." % (
                type(other_sampler),))
        self.other_sampler = other_sampler
        self.p_drop = iap.handle_probability_param(
            p_drop, "p_drop", tuple_to_uniform=True, list_to_choice=True)

    def sample_points(self, imgs, rs):
        rs = iarandom.RNG.create_if_not_rng_(rs)
        _verify_sample_points_images(imgs)

        rss = rs.duplicate(len(imgs) + 1)
        points_imgs = self.other_sampler.sample_points(
            imgs, rss[-1])
        masks = [self._draw_samples_for_img(points_img, rs)
                      for points_img, rs
                      in zip(points_imgs, rss)]
        return self._apply_dropout_masks(points_imgs, masks)

    def _draw_samples_for_img(self, points_img, rs):
        samples = self.p_drop.draw_samples((len(points_img),),
                                                rs)
        mask = (samples > 0.5)
        return mask

    @classmethod
    def _apply_dropout_masks(cls, points_imgs, masks):
        points_imgs_masked = []
        for points_img, mask in zip(points_imgs, masks):
            if len(points_img) == 0:
                # other sampler didn't provide any points
                points_masked = points_img
            else:
                if not np.any(mask):
                    # keep at least one point if all were supposed to be
                    # dropped
                    # TODO this could also be moved into its own point sampler,
                    #      like AtLeastOnePoint(...)
                    idx = (len(points_img) - 1) // 2
                    mask = np.copy(mask)
                    mask[idx] = True
                points_masked = points_img[mask, :]
            points_imgs_masked.append(points_masked)
        return points_imgs_masked

    def __repr__(self):
        return "DropoutPointsSampler(%s, %s)" % (self.other_sampler,
                                                 self.p_drop)

    def __str__(self):
        return self.__repr__()


class UniformPointsSampler(IPointsSampler):
    """Sample points uniformly on images.

    This point sampler generates `n_points` points per image. The x- and
    y-coordinates are both sampled from uniform distributions matching the
    respective image width and height.

    Parameters
    ----------
    n_points : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of points to sample on each image.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.UniformPointsSampler(500)

    Create a point sampler that generates an array of ``500`` random points for
    each input image. The x- and y-coordinates of each point are sampled
    from uniform distributions.

    """

    def __init__(self, n_points):
        self.n_points = iap.handle_discrete_param(
            n_points, "n_points", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)

    def sample_points(self, imgs, rs):
        rs = iarandom.RNG.create_if_not_rng_(rs)
        _verify_sample_points_images(imgs)

        rss = rs.duplicate(2)
        n_points_imgs = self._draw_samples(len(imgs), rss[0])

        n_points_total = np.sum(n_points_imgs)
        n_total = 2 * n_points_total
        coords_rel = rss[1].uniform(0.0, 1.0, n_total)
        coords_rel_xy = coords_rel.reshape(n_points_total, 2)

        return self._convert_rel_coords_to_abs(
            coords_rel_xy, n_points_imgs, imgs)

    def _draw_samples(self, n_imgs, rs):
        n_points = self.n_points.draw_samples((n_imgs,),
                                              random_state=rs)
        n_points = np.clip(n_points, 1, None)
        return n_points

    @classmethod
    def _convert_rel_coords_to_abs(cls, coords_rel_xy,
                                             n_points_imgs, imgs):
        coords_abs = []
        i = 0
        for img, n_points_img in zip(imgs, n_points_imgs):
            h, w = img.shape[0:2]
            x = coords_rel_xy[i:i+n_points_img, 0]
            y = coords_rel_xy[i:i+n_points_img, 1]

            x_int = np.clip(np.round(x * w), 0, w)
            y_int = np.clip(np.round(y * h), 0, h)

            coords_abs.append(np.stack([x_int, y_int], axis=-1))
            i += n_points_img
        return coords_abs

    def __repr__(self):
        return "UniformPointsSampler(%s)" % (self.n_points,)

    def __str__(self):
        return self.__repr__()


class SubsamplingPointsSampler(IPointsSampler):
    """Ensure that the number of sampled points is below a maximum.

    This point sampler will sample points from another sampler and
    then -- in case more points were generated than an allowed maximum --
    will randomly pick `n_points_max` of these.

    Parameters
    ----------
    other_sampler : IPointsSampler
        Another point sampler that is queried to generate a ``list`` of points.
        The dropout operation will be applied to that ``list``.

    n_points_max : int
        Maximum number of allowed points. If `other_sampler` generates
        more points than this maximum, a random subset of size `n_points_max`
        will be selected.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.SubsamplingPointsSampler(
    >>>     iaa.RelativeRegularGridPointsSampler(0.1, 0.2),
    >>>     50
    >>> )

    Create a points sampler that places ``y*H`` points on the y-axis (with
    ``y`` being ``0.1`` and ``H`` being an image's height) and ``x*W`` on
    the x-axis (analogous). Then, if that number of placed points exceeds
    ``50`` (can easily happen for larger images), a random subset of ``50``
    points will be picked and returned.

    """

    def __init__(self, other_sampler, n_points_max):
        assert isinstance(other_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            "'other_sampler', got type %s." % (
                type(other_sampler),))
        self.other_sampler = other_sampler
        self.n_points_max = np.clip(n_points_max, -1, None)
        if self.n_points_max == 0:
            ia.warn("Got n_points_max=0 in SubsamplingPointsSampler. "
                    "This will result in no points ever getting "
                    "returned.")

    def sample_points(self, imgs, rs):
        rs = iarandom.RNG.create_if_not_rng_(rs)
        _verify_sample_points_images(imgs)

        rss = rs.duplicate(len(imgs) + 1)
        points_imgs = self.other_sampler.sample_points(
            imgs, rss[-1])
        return [self._subsample(points_img, self.n_points_max, rs)
                for points_img, rs
                in zip(points_imgs, rss[:-1])]

    @classmethod
    def _subsample(cls, points_img, n_points_max, rs):
        if len(points_img) <= n_points_max:
            return points_img
        indices = np.arange(len(points_img))
        indices_keep = rs.permutation(indices)[0:n_points_max]
        return points_img[indices_keep]

    def __repr__(self):
        return "SubsamplingPointsSampler(%s, %d)" % (self.other_sampler,
                                                     self.n_points_max)

    def __str__(self):
        return self.__repr__()


# TODO Add points subsampler that drops points close to each other first
# TODO Add poisson points sampler
# TODO Add jitter points sampler that moves points around
# for both see https://codegolf.stackexchange.com/questions/50299/draw-an-image-as-a-voronoi-map/50345#50345