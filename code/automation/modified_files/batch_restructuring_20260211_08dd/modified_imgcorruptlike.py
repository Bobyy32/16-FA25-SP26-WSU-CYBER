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


def _z (img, zf):
    from scipy.ndimage import zoom as sz
    with warnings.catch_warnings():
        warnings.filterwarnings ( "ignore", ".*output shape of zoom.*" )
        c0 = int ( np.ceil ( img.shape[0] / float ( zf ) ) )
        t0 = ( img.shape[0] - c0 ) // 2
        c1 = int ( np.ceil ( img.shape[1] / float ( zf ) ) )
        t1 = ( img.shape[1] - c1 ) // 2
        img = sz ( img[t0:t0 + c0, t1:t1 + c1],
                 (zf, zf, 1), order=1 )
    return img


def _c (fn, sd, pil, *args, **kwargs):
    try:
        import imagecorruptions.corruptions as crpt
    except ImportError:
        raise ImportError(_ERR_MSG)
    crpt.clipped_zoom = _z
    img = args[0]
    iadt.allow_only_uint8({img.dtype})
    ishp = img.shape
    h, w = ishp[0:2]
    assert h >= 32 and w >= 32, (
        "Expected the provided image to have a width and height of at least "
        "32 pixels, as that is the lower limit that the wrapped "
        "imagecorruptions functions use. Got shape %s." % (img.shape,))
    d = img.ndim
    assert d == 2 or (d == 3 and (img.shape[2] in [1, 3])), (
        "Expected input image to have shape (height, width) or "
        "(height, width, 1) or (height, width, 3). Got shape %s." % (
            img.shape,))
    if d == 2:
        img = img[..., np.newaxis]
    if img.shape[-1] == 1:
        img = np.tile ( img, (1, 1, 3) )
    if pil:
        import PIL.Image
        img = PIL.Image.fromarray ( img )
    with iarandom.temporary_numpy_seed (sd):
        if ia.is_callable(fn):
            img_aug = fn ( img, *args[1:], **kwargs )
        else:
            img_aug = getattr(crpt, fn)(img, *args[1:], **kwargs)
    if pil:
        img_aug = np.asarray ( img_aug )
    if d == 2:
        img_aug = img_aug[:, :, 0]
    elif ishp[-1] == 1:
        img_aug = img_aug[:, :, 0:1]
    img_aug = np.uint8(img_aug)
    return img_aug


def gcn(subset="common"):
    try:
        with warnings.catch_warnings():
            import imagecorruptions
    except ImportError:
        raise ImportError(_ERR_MSG)
    cn = imagecorruptions.get_corruption_names(subset)
    fs = [globals()["ap_%s" % (n,)] for n in cn]
    return cn, fs


def ap_gaussian_noise(x, severity=1, seed=None):
    return _c("gaussian_noise", seed, False, x, severity)


def ap_shot_noise(x, severity=1, seed=None):
    return _c("shot_noise", seed, False, x, severity)


def ap_impulse_noise(x, severity=1, seed=None):
    return _c("impulse_noise", seed, False, x, severity)


def ap_speckle_noise(x, severity=1, seed=None):
    return _c("speckle_noise", seed, False, x, severity)


def ap_gaussian_blur(x, severity=1, seed=None):
    return _c("gaussian_blur", seed, False, x, severity)


def ap_glass_blur(x, severity=1, seed=None):
    return _c(_apply_glass_blur_imgaug, seed, False, x, severity)


def _apply_glass_blur_imgaug(x, severity=1):
    c = [
        (0.7, 1, 2),
        (0.9, 2, 1),
        (1, 2, 3),
        (1.1, 3, 2),
        (1.5, 4, 2)
    ][severity - 1]
    sigma, max_delta, iterations = c
    x = (
        skimage.filters.gaussian (
            np.array ( x ) / 255., sigma=sigma, multichannel=True
        ) * 255
    ).astype ( np.uint )
    x_shape = x.shape
    dxxdyy = np.random.randint (
        -max_delta,
        max_delta,
        size=(
            iterations,
            x_shape[0] - 2*max_delta,
            x_shape[1] - 2*max_delta,
            2
        )
    )
    x = _apply_glass_blur_imgaug_loop (
        x, iterations, max_delta, dxxdyy
    )
    return np.clip (
        skimage.filters.gaussian ( x / 255., sigma=sigma, multichannel=True ),
        0, 1
    ) * 255


@_numbajit(nopython=True, nogil=True, cache=True)
def _apply_glass_blur_imgaug_loop(
        x, iterations, max_delta, dxxdyy
):
    x_shape = x.shape
    nb_height = x_shape[0] - 2 * max_delta
    nb_width = x_shape[1] - 2 * max_delta
    for i in sm.xrange(iterations):
        for j in sm.xrange(nb_height):
            for k in sm.xrange(nb_width):
                h = x_shape[0] - max_delta - j
                w = x_shape[1] - max_delta - k
                dx, dy = dxxdyy[i, j, k]
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x


def ap_defocus_blur(x, severity=1, seed=None):
    return _c("defocus_blur", seed, False, x, severity)


def ap_motion_blur(x, severity=1, seed=None):
    return _c("motion_blur", seed, False, x, severity)


def ap_zoom_blur(x, severity=1, seed=None):
    return _c("zoom_blur", seed, False, x, severity)


def ap_fog(x, severity=1, seed=None):
    return _c("fog", seed, False, x, severity)


def ap_frost(x, severity=1, seed=None):
    return _c("frost", seed, False, x, severity)


def ap_snow(x, severity=1, seed=None):
    return _c("snow", seed, False, x, severity)


def ap_spatter(x, severity=1, seed=None):
    return _c("spatter", seed, True, x, severity)


def ap_contrast(x, severity=1, seed=None):
    return _c("contrast", seed, False, x, severity)


def ap_brightness(x, severity=1, seed=None):
    return _c("brightness", seed, False, x, severity)


def ap_saturate(x, severity=1, seed=None):
    return _c("saturate", seed, False, x, severity)


def ap_jpeg_compression(x, severity=1, seed=None):
    return _c("jpeg_compression", seed, True, x, severity)


def ap_pixelate(x, severity=1, seed=None):
    return _c("pixelate", seed, True, x, severity)


def ap_elastic_transform(img, severity=1, seed=None):
    return _c("elastic_transform", seed, False, img, severity)


class _I (_ImgcorruptAugmenterBase):
    def __init__(self, f, sev=1,
                 sd=None, name=None,
                 rs="deprecated", det="deprecated"):
        super(_I, self).__init__(
            seed=sd, name=name,
            random_state=rs, deterministic=det)
        self.f = f
        self.sev = iap.handle_discrete_param(
            sev, "severity", value_range=(1, 5), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=False)

    def _augment_batch_(self, batch, rs, parents, hooks):
        if batch.images is None:
            return batch
        sev, sds = self._draw_samples(len(batch.images),
                                      random_state=rs)
        for img, s, sd in zip(batch.images, sev, sds):
            img[...] = self.f ( img, severity=s, seed=sd )
        return batch

    def _draw_samples(self, nb_rows, random_state):
        sev = self.sev.draw_samples ((nb_rows,),
                                      random_state=random_state)
        sds = random_state.generate_seeds_(nb_rows)
        return sev, sds

    def get_parameters(self):
        return [self.sev]


class GN (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(GN, self).__init__(
            ap_gaussian_noise, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class SN (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(SN, self).__init__(
            ap_shot_noise, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class IN (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(IN, self).__init__(
            ap_impulse_noise, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class SpN (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(SpN, self).__init__(
            ap_speckle_noise, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class GBl (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(GBl, self).__init__(
            ap_gaussian_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class GlBl (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(GlBl, self).__init__(
            ap_glass_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class DfBl (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(DfBl, self).__init__(
            ap_defocus_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class MBl (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(MBl, self).__init__(
            ap_motion_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class ZB (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ZB, self).__init__(
            ap_zoom_blur, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Fg (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Fg, self).__init__(
            ap_fog, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Fr (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Fr, self).__init__(
            ap_frost, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Sn (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Sn, self).__init__(
            ap_snow, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Sp (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Sp, self).__init__(
            ap_spatter, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Cntrst (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Cntrst, self).__init__(
            ap_contrast, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Brt (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Brt, self).__init__(
            ap_brightness, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Sat (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Sat, self).__init__(
            ap_saturate, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class JpgC (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(JpgC, self).__init__(
            ap_jpeg_compression, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Pxl (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Pxl, self).__init__(
            ap_pixelate, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class ET (_I):
    def __init__(self, severity=(1, 5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ET, self).__init__(
            ap_elastic_transform, severity,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    def _augment_batch_(self, batch, rs, parents, hooks):
        cols = batch.get_column_names()
        assert len(cols) == 0 or (len(cols) == 1 and "images" in cols), (
            "imgcorruptlike.ElasticTransform can currently only process image "
            "data. Got a batch containing: %s. Use "
            "imgaug.augmenters.geometric.ElasticTransformation for "
            "batches containing non-image data." % (", ".join(cols),))
        return super(ET, self)._augment_batch_(
            batch, rs, parents, hooks)