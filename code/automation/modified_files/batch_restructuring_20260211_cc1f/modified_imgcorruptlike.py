from __future__ import print_function, division, absolute_import

import warnings

import six.moves as sm
import numpy as np
import skimage.filters

import imgaug as ia
from ..imgaug import _numbajit
from .. import dtypes as iadt
from .. import random as iarandom
from .. import parameters as iap
from . import meta

_ERR_MSG = (
    "Could not import package `imagecorruptions`. This is an optional "
    "dependency of imgaug and must be installed manually in order "
    "to use augmenters from `imgaug.augmenters.imgcorrupt`. "
    "Use e.g. `pip install imagecorruptions` to install it. See also "
    "https://github.com/bethgelab/imagecorruptions for the repository "
    "of the package."
)


def _clip_zoom_warn(img, zf):
    from scipy.ndimage import zoom as scizoom

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*output shape of zoom.*")

        d0 = int(np.ceil(img.shape[0] / float(zf)))
        t0 = (img.shape[0] - d0) // 2

        d1 = int(np.ceil(img.shape[1] / float(zf)))
        t1 = (img.shape[1] - d1) // 2

        img = scizoom(img[t0:t0 + d0, t1:t1 + d1],
                      (zf, zf, 1), order=1)

        return img


def _call_icorrupt(fname, seed, pil_conv, *args, **kwargs):
    """Apply an 'imagecorruptions' function."""
    try:
        with warnings.catch_warnings():
            import imagecorruptions.corruptions as corr_funcs
    except ImportError:
        raise ImportError(_ERR_MSG)

    corr_funcs.clipped_zoom = _clip_zoom_warn

    img_input = args[0]

    iadt.allow_only_uint8({img_input.dtype})

    inp_shape = img_input.shape

    h, w = inp_shape[0:2]
    assert h >= 32 and w >= 32, (
        "Expected image H/W >= 32. Got shape %s." % (img_input.shape,))

    ndims = img_input.ndim
    assert ndims == 2 or (ndims == 3 and (img_input.shape[2] in [1, 3])), (
        "Expected input image shape (H,W) or (H,W,1) or (H,W,3). Got shape %s." % (
            img_input.shape,))

    if ndims == 2:
        img_input = img_input[..., np.newaxis]
    if img_input.shape[-1] == 1:
        img_input = np.tile(img_input, (1, 1, 3))

    if pil_conv:
        import PIL.Image
        img_input = PIL.Image.fromarray(img_input)

    with iarandom.temporary_numpy_seed(seed):
        if ia.is_callable(fname):
            img_out = fname(img_input, *args[1:], **kwargs)
        else:
            img_out = getattr(corr_funcs, fname)(img_input, *args[1:], **kwargs)

    if pil_conv:
        img_out = np.asarray(img_out)

    if ndims == 2:
        img_out = img_out[:, :, 0]
    elif inp_shape[-1] == 1:
        img_out = img_out[:, :, 0:1]

    img_out = np.uint8(img_out)

    return img_out


def get_corruption_names(subset="common"):
    """Get a named subset of corruption functions."""
    try:
        with warnings.catch_warnings():
            import imagecorruptions
    except ImportError:
        raise ImportError(_ERR_MSG)

    cnames = imagecorruptions.get_corruption_names(subset)
    fnames = [globals()["apply_%s" % (cname,)] for cname in cnames]

    return cnames, fnames


def _apply_gaussian_noise(x, severity=1, seed=None):
    """Apply `gaussian_noise`."""
    return _call_icorrupt("gaussian_noise", seed, False, x, severity)


def _apply_shot_noise(x, severity=1, seed=None):
    """Apply `shot_noise`."""
    return _call_icorrupt("shot_noise", seed, False, x, severity)


def _apply_impulse_noise(x, severity=1, seed=None):
    """Apply `impulse_noise`."""
    return _call_icorrupt("impulse_noise", seed, False, x, severity)


def _apply_speckle_noise(x, severity=1, seed=None):
    """Apply `speckle_noise`."""
    return _call_icorrupt("speckle_noise", seed, False, x, severity)


def _apply_gaussian_blur(x, severity=1, seed=None):
    """Apply `gaussian_blur`."""
    return _call_icorrupt("gaussian_blur", seed, False, x, severity)


def _apply_glass_blur(x, severity=1, seed=None):
    """Apply `glass_blur`."""
    return _call_icorrupt(_apply_glass_blur_imgaug, seed, False, x, severity)


def _apply_glass_blur_imgaug(x, severity=1):
    """Internal implementation for glass blur."""
    c = [
        (0.7, 1, 2),
        (0.9, 2, 1),
        (1, 2, 3),
        (1.1, 3, 2),
        (1.5, 4, 2)
    ][severity - 1]

    sigma, max_delta, iterations = c

    x = (
        skimage.filters.gaussian(
            np.array(x) / 255., sigma=sigma, multichannel=True
        ) * 255
    ).astype(np.uint)
    x_shape = x.shape

    dxdy = np.random.randint(
        -max_delta,
        max_delta,
        size=(
            iterations,
            x_shape[0] - 2*max_delta,
            x_shape[1] - 2*max_delta,
            2
        )
    )

    x = _apply_glass_blur_imgaug_loop(
        x, iterations, max_delta, dxdy
    )

    return np.clip(
        skimage.filters.gaussian(x / 255., sigma=sigma, multichannel=True),
        0, 1
    ) * 255


@_numbajit(nopython=True, nogil=True, cache=True)
def _apply_glass_blur_imgaug_loop(
        x, iterations, max_delta, dxdy
):
    """Numba-accelerated loop for glass blur."""
    x_shape = x.shape
    nb_h = x_shape[0] - 2 * max_delta
    nb_w = x_shape[1] - 2 * max_delta

    for i in sm.xrange(iterations):
        for j in sm.xrange(nb_h):
            for k in sm.xrange(nb_w):
                h = x_shape[0] - max_delta - j
                w = x_shape[1] - max_delta - k
                dx, dy = dxdy[i, j, k]
                h_p, w_p = h + dy, w + dx
                x[h, w], x[h_p, w_p] = x[h_p, w_p], x[h, w]

    return x


def _apply_defocus_blur(x, severity=1, seed=None):
    """Apply `defocus_blur`."""
    return _call_icorrupt("defocus_blur", seed, False, x, severity)


def _apply_motion_blur(x, severity=1, seed=None):
    """Apply `motion_blur`."""
    return _call_icorrupt("motion_blur", seed, False, x, severity)


def _apply_zoom_blur(x, severity=1, seed=None):
    """Apply `zoom_blur`."""
    return _call_icorrupt("zoom_blur", seed, False, x, severity)


def _apply_fog(x, severity=1, seed=None):
    """Apply `fog`."""
    return _call_icorrupt("fog", seed, False, x, severity)


def _apply_frost(x, severity=1, seed=None):
    """Apply `frost`."""
    return _call_icorrupt("frost", seed, False, x, severity)


def _apply_snow(x, severity=1, seed=None):
    """Apply `snow`."""
    return _call_icorrupt("snow", seed, False, x, severity)


def _apply_spatter(x, severity=1, seed=None):
    """Apply `spatter`."""
    return _call_icorrupt("spatter", seed, True, x, severity)


def _apply_contrast(x, severity=1, seed=None):
    """Apply `contrast`."""
    return _call_icorrupt("contrast", seed, False, x, severity)


def _apply_brightness(x, severity=1, seed=None):
    """Apply `brightness`."""
    return _call_icorrupt("brightness", seed, False, x, severity)


def _apply_saturate(x, severity=1, seed=None):
    """Apply `saturate`."""
    return _call_icorrupt("saturate", seed, False, x, severity)


def _apply_jpeg_compression(x, severity=1, seed=None):
    """Apply `jpeg_compression`."""
    return _call_icorrupt("jpeg_compression", seed, True, x, severity)


def _apply_pixelate(x, severity=1, seed=None):
    """Apply `pixelate`."""
    return _call_icorrupt("pixelate", seed, True, x, severity)


def _apply_elastic_transform(img_data, severity=1, seed=None):
    """Apply `elastic_transform`."""
    return _call_icorrupt("elastic_transform", seed, False, img_data, severity)


class _BaseAug(meta.Augmenter):
    def __init__(self, func, severity=1, seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(_BaseAug, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.func = func
        self.severity = iap.handle_discrete_param(
            severity, "severity", value_range=(1, 5), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=False)

    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        s, ss = self._draw_samples(len(batch.images),
                                   random_state=random_state)

        for idx, (img, sev, sd) in enumerate(zip(batch.images, s, ss)):
            batch.images[idx] = self.func(img, severity=sev, seed=sd)

        return batch

    def _draw_samples(self, n_rows, random_state):
        samps_sev = self.severity.draw_samples((n_rows,),
                                               random_state=random_state)
        samps_seed = random_state.generate_seeds_(n_rows)

        return samps_sev, samps_seed

    def get_parameters(self):
        """Return parameters."""
        return [self.severity]


class GaussianNoise(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.gaussian_noise`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(GaussianNoise, self).__init__(
            _apply_gaussian_noise, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class ShotNoise(_BaseAug):
    """Wrapper for `imagecorruptions.shot_noise`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ShotNoise, self).__init__(
            _apply_shot_noise, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class ImpulseNoise(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.impulse_noise`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ImpulseNoise, self).__init__(
            _apply_impulse_noise, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class SpeckleNoise(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.speckle_noise`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(SpeckleNoise, self).__init__(
            _apply_speckle_noise, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class GaussianBlur(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.gaussian_blur`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(GaussianBlur, self).__init__(
            _apply_gaussian_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class GlassBlur(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.glass_blur`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(GlassBlur, self).__init__(
            _apply_glass_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class DefocusBlur(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.defocus_blur`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(DefocusBlur, self).__init__(
            _apply_defocus_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class MotionBlur(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.motion_blur`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(MotionBlur, self).__init__(
            _apply_motion_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class ZoomBlur(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.zoom_blur`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ZoomBlur, self).__init__(
            _apply_zoom_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Fog(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.fog`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Fog, self).__init__(
            _apply_fog, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Frost(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.frost`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Frost, self).__init__(
            _apply_frost, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Snow(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.snow`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Snow, self).__init__(
            _apply_snow, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Spatter(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.spatter`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Spatter, self).__init__(
            _apply_spatter, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Contrast(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.contrast`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Contrast, self).__init__(
            _apply_contrast, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Brightness(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.brightness`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Brightness, self).__init__(
            _apply_brightness, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Saturate(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.saturate`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Saturate, self).__init__(
            _apply_saturate, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class JpegCompression(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.jpeg_compression`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(JpegCompression, self).__init__(
            _apply_jpeg_compression, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Pixelate(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.pixelate`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Pixelate, self).__init__(
            _apply_pixelate, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class ElasticTransform(_BaseAug):
    """Wrapper for `imagecorruptions.corruptions.elastic_transform`."""

    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ElasticTransform, self).__init__(
            _apply_elastic_transform, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    def _augment_batch_(self, batch, random_state, parents, hooks):
        cols = batch.get_column_names()
        assert len(cols) == 0 or (len(cols) == 1 and "images" in cols), (
            "imgcorruptlike.ElasticTransform can currently only process image "
            "data. Got a batch containing: %s. Use "
            "imgaug.augmenters.geometric.ElasticTransformation for "
            "batches containing non-image data." % (", ".join(cols),))
        return super(ElasticTransform, self)._augment_batch_(
            batch, random_state, parents, hooks)