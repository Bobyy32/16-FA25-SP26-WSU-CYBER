"""Functions to generate example data, e.g. example images or segmaps.

Added in 0.5.0.

"""
from __future__ import print_function, division, absolute_import

import os
import json

import imageio
import numpy as np

# filepath to the quokka image, its annotations and depth map
# Added in 0.5.0.
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# Added in 0.5.0.
_QUOKKA_FP = os.path.join(_FILE_DIR, "quokka.jpg")
# Added in 0.5.0.
_QUOKKA_ANNOTATIONS_FP = os.path.join(_FILE_DIR, "quokka_annotations.json")
# Added in 0.5.0.
_QUOKKA_DEPTH_MAP_HALFRES_FP = os.path.join(
    _FILE_DIR, "quokka_depth_map_halfres.png")


def _extract_normalized_bounding_box(extract_input):
    """Generate a normalized rectangle for the standard quokka image.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    extract_input : 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Unnormalized representation of the image subarea to be extracted.

            * If ``str`` ``square``, then a squared area
              ``(x: 0 to max 643, y: 0 to max 643)`` will be extracted from
              the image.
            * If a ``tuple``, then expected to contain four ``number`` s
              denoting ``(x1, y1, x2, y2)``.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBox`, then that
              bounding box's area will be extracted from the image.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBoxesOnImage`,
              then expected to contain exactly one bounding box and a shape
              matching the full image dimensions (i.e. ``(643, 960, *)``).
              Then the one bounding box will be used similar to
              ``BoundingBox`` above.

    Returns
    -------
    imgaug.augmentables.bbs.BoundingBox
        Normalized representation of the area to extract from the standard
        quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

    if extract_input == "square":
        bounding_box = BoundingBox(x1=0, y1=0, x2=643, y2=643)
    elif isinstance(extract_input, tuple) and len(extract_input) == 4:
        bounding_box = BoundingBox(x1=extract_input[0], y1=extract_input[1],
                                   x2=extract_input[2], y2=extract_input[3])
    elif isinstance(extract_input, BoundingBox):
        bounding_box = extract_input
    elif isinstance(extract_input, BoundingBoxesOnImage):
        assert len(extract_input.bounding_boxes) == 1, (
            "Provided BoundingBoxesOnImage instance may currently only "
            "contain a single bounding box.")
        assert extract_input.shape[0:2] == (643, 960), (
            "Expected BoundingBoxesOnImage instance on an image of shape "
            "(643, 960, ?). Got shape %s." % (extract_input.shape,))
        bounding_box = extract_input.bounding_boxes[0]
    else:
        raise Exception(
            "Expected 'square' or tuple of four entries or BoundingBox or "
            "BoundingBoxesOnImage for parameter 'extract', "
            "got %s." % (type(extract_input),)
        )
    return bounding_box


def _compute_new_shape(original_shape, target_shape):
    """Compute the intended new shape of an image-like array after resizing.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    original_shape : tuple or ndarray
        Old shape of the array. Usually expected to be a ``tuple`` of form
        ``(H, W)`` or ``(H, W, C)`` or alternatively an array with two or
        three dimensions.

    target_shape : None or tuple of ints or tuple of floats or int or float or ndarray
        New shape of the array.

            * If ``None``, then `original_shape` will be used as the new shape.
            * If an ``int`` ``V``, then the new shape will be ``(V, V, [C])``,
              where ``C`` will be added if it is part of `original_shape`.
            * If a ``float`` ``V``, then the new shape will be
              ``(H*V, W*V, [C])``, where ``H`` and ``W`` are the old
              height/width.
            * If a ``tuple`` ``(H', W', [C'])`` of ints, then ``H'`` and ``W'``
              will be used as the new height and width.
            * If a ``tuple`` ``(H', W', [C'])`` of floats (except ``C``), then
              ``H'`` and ``W'`` will be used as the new height and width.
            * If a numpy array, then the array's shape will be used.

    Returns
    -------
    tuple of int
        New shape.

    """
    from . import imgaug as ia

    if ia.is_np_array(original_shape):
        original_shape = original_shape.shape
    if ia.is_np_array(target_shape):
        target_shape = target_shape.shape

    computed_shape = list(original_shape)

    if target_shape is None:
        pass
    elif isinstance(target_shape, tuple):
        assert len(original_shape) in [2, 3]
        assert len(target_shape) in [2, 3]

        if len(original_shape) == 3 and len(target_shape) == 3:
            assert original_shape[2] == target_shape[2]
        elif len(target_shape) == 3:
            computed_shape.append(target_shape[2])

        is_target_shape_valid_values = all(
            [v is None or ia.is_single_number(v) for v in target_shape[0:2]])
        assert is_target_shape_valid_values, (
            "Expected the first two entries in target_shape to be None or "
            "numbers, got types %s." % (
                str([type(v) for v in target_shape[0:2]]),))

        for i, original_dim in enumerate(original_shape[0:2]):
            if target_shape[i] is None:
                computed_shape[i] = original_dim
            elif ia.is_single_integer(target_shape[i]):
                computed_shape[i] = target_shape[i]
            else:  # float
                computed_shape[i] = int(np.round(original_dim * target_shape[i]))
    elif ia.is_single_integer(target_shape) or ia.is_single_float(target_shape):
        computed_shape = _compute_new_shape(
            original_shape, (target_shape, target_shape))
    else:
        raise Exception(
            "Expected target_shape to be None or ndarray or tuple of floats or "
            "tuple of ints or single int or single float, "
            "got %s." % (type(target_shape),))

    return tuple(computed_shape)


def quokka(size=None, extract=None):
    """Return an image of a quokka as a numpy array.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int, optional
        Size of the output image. Input into
        :func:`~imgaug.imgaug.imresize_single_image`. Usually expected to be a
        ``tuple`` ``(H, W)``, where ``H`` is the desired height and ``W`` is
        the width. If ``None``, then the image will not be resized.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea of the quokka image to extract:

            * If ``None``, then the whole image will be used.
            * If ``str`` ``square``, then a squared area
              ``(x: 0 to max 643, y: 0 to max 643)`` will be extracted from
              the image.
            * If a ``tuple``, then expected to contain four ``number`` s
              denoting ``(x1, y1, x2, y2)``.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBox`, then that
              bounding box's area will be extracted from the image.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBoxesOnImage`,
              then expected to contain exactly one bounding box and a shape
              matching the full image dimensions (i.e. ``(643, 960, *)``).
              Then the one bounding box will be used similar to
              ``BoundingBox`` above.

    Returns
    -------
    (H,W,3) ndarray
        The image array of dtype ``uint8``.

    """
    from . import imgaug as ia

    image_data = imageio.imread(_QUOKKA_FP, pilmode="RGB")
    if extract is not None:
        normalized_bounding_box = _extract_normalized_bounding_box(extract)
        image_data = normalized_bounding_box.extract_from_image(image_data)
    if size is not None:
        resized_shape = _compute_new_shape(image_data.shape, size)
        image_data = ia.imresize_single_image(image_data, resized_shape[0:2])
    return image_data


def quokka_square(size=None):
    """Return an (square) image of a quokka as a numpy array.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int, optional
        Size of the output image. Input into
        :func:`~imgaug.imgaug.imresize_single_image`. Usually expected to be a
        ``tuple`` ``(H, W)``, where ``H`` is the desired height and ``W`` is
        the width. If ``None``, then the image will not be resized.

    Returns
    -------
    (H,W,3) ndarray
        The image array of dtype ``uint8``.

    """
    return quokka(size=size, extract="square")


def quokka_heatmap(size=None, extract=None):
    """Return a heatmap (here: depth map) for the standard example quokka image.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int, optional
        See :func:`~imgaug.imgaug.quokka`.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.heatmaps.HeatmapsOnImage
        Depth map as an heatmap object. Values close to ``0.0`` denote objects
        that are close to the camera. Values close to ``1.0`` denote objects
        that are furthest away (among all shown objects).

    """
    # TODO get rid of this deferred import
    from . import imgaug as ia
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

    image_data = imageio.imread(_QUOKKA_DEPTH_MAP_HALFRES_FP, pilmode="RGB")
    image_data = ia.imresize_single_image(image_data, (643, 960), interpolation="cubic")

    if extract is not None:
        normalized_bounding_box = _extract_normalized_bounding_box(extract)
        image_data = normalized_bounding_box.extract_from_image(image_data)
    if size is None:
        size = image_data.shape[0:2]

    resized_shape = _compute_new_shape(image_data.shape, size)
    image_data = ia.imresize_single_image(image_data, resized_shape[0:2])
    channel_0_data = image_data[..., 0]  # depth map was saved as 3-channel RGB
    channel_0_data = channel_0_data.astype(np.float32) / 255.0
    channel_0_data = 1 - channel_0_data  # depth map was saved as 0 being furthest away

    return HeatmapsOnImage(channel_0_data, shape=channel_0_data.shape[0:2] + (3,))


def quokka_segmentation_map(size=None, extract=None):
    """Return a segmentation map for the standard example quokka image.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int, optional
        See :func:`~imgaug.imgaug.quokka`.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.segmaps.SegmentationMapsOnImage
        Segmentation map object.

    """
    # pylint: disable=invalid-name
    import skimage.draw
    # TODO get rid of this deferred import
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    with open(_QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)

    x_coords = []
    y_coords = []
    for kp_dict in json_dict["polygons"][0]["keypoints"]:
        x_coords.append(kp_dict["x"])
        y_coords.append(kp_dict["y"])

    segmentation_image = np.zeros((643, 960, 1), dtype=np.int32)
    rr, cc = skimage.draw.polygon(
        np.array(y_coords), np.array(x_coords), shape=segmentation_image.shape)
    segmentation_image[rr, cc, 0] = 1

    if extract is not None:
        normalized_bounding_box = _extract_normalized_bounding_box(extract)
        segmentation_image = normalized_bounding_box.extract_from_image(segmentation_image)

    segmentation_map = SegmentationMapsOnImage(segmentation_image, shape=segmentation_image.shape[0:2] + (3,))

    if size is not None:
        resized_shape = _compute_new_shape(segmentation_image.shape, size)
        segmentation_map = segmentation_map.resize(resized_shape[0:2])
        segmentation_map.shape = tuple(resized_shape[0:2]) + (3,)

    return segmentation_map


def quokka_keypoints(size=None, extract=None):
    """Return example keypoints on the standard example quokke image.

    The keypoints cover the eyes, ears, nose and paws.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the keypoints are placed. If
        ``None``, then the keypoints are not projected to any new size
        (positions on the original image are used). ``float`` s lead to
        relative size changes, ``int`` s to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.kps.KeypointsOnImage
        Example keypoints on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

    offset_x, offset_y = 0, 0
    if extract is not None:
        normalized_bounding_box = _extract_normalized_bounding_box(extract)
        offset_x = normalized_bounding_box.x1
        offset_y = normalized_bounding_box.y1
    with open(_QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    keypoints_list = []
    for kp_dict in json_dict["keypoints"]:
        keypoints_list.append(Keypoint(x=kp_dict["x"] - offset_x, y=kp_dict["y"] - offset_y))
    if extract is not None:
        current_shape = (normalized_bounding_box.height, normalized_bounding_box.width, 3)
    else:
        current_shape = (643, 960, 3)
    keypoints_on_image = KeypointsOnImage(keypoints_list, shape=current_shape)
    if size is not None:
        resized_shape = _compute_new_shape(current_shape, size)
        keypoints_on_image = keypoints_on_image.on(resized_shape)
    return keypoints_on_image


def quokka_bounding_boxes(size=None, extract=None):
    """Return example bounding boxes on the standard example quokke image.

    Currently only a single bounding box is returned that covers the quokka.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the BBs are placed. If ``None``, then
        the BBs are not projected to any new size (positions on the original
        image are used). ``float`` s lead to relative size changes, ``int`` s
        to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.bbs.BoundingBoxesOnImage
        Example BBs on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

    offset_x, offset_y = 0, 0
    if extract is not None:
        normalized_bounding_box = _extract_normalized_bounding_box(extract)
        offset_x = normalized_bounding_box.x1
        offset_y = normalized_bounding_box.y1
    with open(_QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    bounding_boxes_list = []
    for bb_dict in json_dict["bounding_boxes"]:
        bounding_boxes_list.append(
            BoundingBox(
                x1=bb_dict["x1"] - offset_x,
                y1=bb_dict["y1"] - offset_y,
                x2=bb_dict["x2"] - offset_x,
                y2=bb_dict["y2"] - offset_y
            )
        )
    if extract is not None:
        current_shape = (normalized_bounding_box.height, normalized_bounding_box.width, 3)
    else:
        current_shape = (643, 960, 3)
    bounding_boxes_on_image = BoundingBoxesOnImage(bounding_boxes_list, shape=current_shape)
    if size is not None:
        resized_shape = _compute_new_shape(current_shape, size)
        bounding_boxes_on_image = bounding_boxes_on_image.on(resized_shape)
    return bounding_boxes_on_image


def quokka_polygons(size=None, extract=None):
    """
    Returns example polygons on the standard example quokke image.

    The result contains one polygon, covering the quokka's outline.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the polygons are placed. If ``None``,
        then the polygons are not projected to any new size (positions on the
        original image are used). ``float`` s lead to relative size changes,
        ``int`` s to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.polys.PolygonsOnImage
        Example polygons on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.polys import Polygon, PolygonsOnImage

    offset_x, offset_y = 0, 0
    if extract is not None:
        normalized_bounding_box = _extract_normalized_bounding_box(extract)
        offset_x = normalized_bounding_box.x1
        offset_y = normalized_bounding_box.y1
    with open(_QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    polygons_list = []
    for poly_json in json_dict["polygons"]:
        polygons_list.append(
            Polygon([(point["x"] - offset_x, point["y"] - offset_y)
                     for point in poly_json["keypoints"]])
        )
    if extract is not None:
        current_shape = (normalized_bounding_box.height, normalized_bounding_box.width, 3)
    else:
        current_shape = (643, 960, 3)
    polygons_on_image = PolygonsOnImage(polygons_list, shape=current_shape)
    if size is not None:
        resized_shape = _compute_new_shape(current_shape, size)
        polygons_on_image = polygons_on_image.on(resized_shape)
    return polygons_on_image