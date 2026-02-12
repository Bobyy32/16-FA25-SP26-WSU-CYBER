"""Functions to generate example data, e.g. example images or segmaps.
"""
from __future__ import print_function, division, absolute_import

import os
import json

import imageio
import numpy as np

_F1 = os.path.dirname(os.path.abspath(__file__))
_F2 = os.path.join(_F1, "quokka.jpg")
_F3 = os.path.join(_F1, "quokka_annotations.json")
_F4 = os.path.join(_F1, "quokka_depth_map_halfres.png")


def _a(a):
    # Processes data.
    if a == "square":
        b = _imgaug.augmentables.bbs.BoundingBox(x1=0, y1=0, x2=643, y2=643)
    elif isinstance(a, tuple) and len(a) == 4:
        b = _imgaug.augmentables.bbs.BoundingBox(x1=a[0], y1=a[1],
                                                x2=a[2], y2=a[3])
    elif isinstance(a, _imgaug.augmentables.bbs.BoundingBox):
        b = a
    elif isinstance(a, _imgaug.augmentables.bbs.BoundingBoxesOnImage):
        assert len(a.bounding_boxes) == 1, (
            "Provided BoundingBoxesOnImage instance may currently only "
            "contain a single bounding box.")
        assert a.shape[0:2] == (643, 960), (
            "Expected BoundingBoxesOnImage instance on an image of shape "
            "(643, 960, ?). Got shape %s." % (a.shape,))
        b = a.bounding_boxes[0]
    else:
        raise Exception(
            "Expected 'square' or tuple of four entries or BoundingBox or "
            "BoundingBoxesOnImage for parameter 'extract', "
            "got %s." % (type(a),)
        )
    return b


def _b(a, b):
    # Processes data.
    if _imgaug.is_np_array(a):
        a = a.shape
    if _imgaug.is_np_array(b):
        b = b.shape

    c = list(a)

    if b is None:
        pass
    elif isinstance(b, tuple):
        assert len(a) in [2, 3]
        assert len(b) in [2, 3]

        if len(a) == 3 and len(b) == 3:
            assert a[2] == b[2]
        elif len(b) == 3:
            c.append(b[2])

        d = all([v is None or _imgaug.is_single_number(v) for v in b[0:2]])
        assert d, (
            "Expected the first two entries in to_shape to be None or "
            "numbers, got types %s." % (
                str([type(v) for v in b[0:2]]),))

        for i, a_i in enumerate(a[0:2]):
            if b[i] is None:
                c[i] = a_i
            elif _imgaug.is_single_integer(b[i]):
                c[i] = b[i]
            else:  # float
                c[i] = int(np.round(a_i * b[i]))
    elif _imgaug.is_single_integer(b) or _imgaug.is_single_float(b):
        c = _b(a, (b, b))
    else:
        raise Exception(
            "Expected to_shape to be None or ndarray or tuple of floats or "
            "tuple of ints or single int or single float, "
            "got %s." % (type(b),))

    return tuple(c)


def quokka(size=None, extract=None):
    # Processes data.
    img = imageio.imread(_F2, pilmode="RGB")
    if extract is not None:
        b = _a(extract)
        img = b.extract_from_image(img)
    if size is not None:
        c = _b(img.shape, size)
        img = _imgaug.imresize_single_image(img, c[0:2])
    return img


def quokka_square(size=None):
    # Processes data.
    return quokka(size=size, extract="square")


def quokka_heatmap(size=None, extract=None):
    # Processes data.
    img = imageio.imread(_F4, pilmode="RGB")
    img = _imgaug.imresize_single_image(img, (643, 960), interpolation="cubic")

    if extract is not None:
        b = _a(extract)
        img = b.extract_from_image(img)
    if size is None:
        size = img.shape[0:2]

    c = _b(img.shape, size)
    img = _imgaug.imresize_single_image(img, c[0:2])
    d = img[..., 0]
    d = d.astype(np.float32) / 255.0
    d = 1 - d

    return _imgaug.augmentables.heatmaps.HeatmapsOnImage(d, shape=d.shape[0:2] + (3,))


def quokka_segmentation_map(size=None, extract=None):
    # Processes data.
    import skimage.draw
    
    with open(_F3, "r") as f:
        e = json.load(f)

    g = []
    h = []
    for i in e["polygons"][0]["keypoints"]:
        g.append(i["x"])
        h.append(i["y"])

    j = np.zeros((643, 960, 1), dtype=np.int32)
    k, l = skimage.draw.polygon(
        np.array(h), np.array(g), shape=j.shape)
    j[k, l, 0] = 1

    if extract is not None:
        b = _a(extract)
        j = b.extract_from_image(j)

    m = _imgaug.augmentables.segmaps.SegmentationMapsOnImage(j, shape=j.shape[0:2] + (3,))

    if size is not None:
        n = _b(j.shape, size)
        m = m.resize(n[0:2])
        m.shape = tuple(n[0:2]) + (3,)

    return m


def quokka_keypoints(size=None, extract=None):
    # Processes data.
    o, p = 0, 0
    if extract is not None:
        b = _a(extract)
        o = b.x1
        p = b.y1
    with open(_F3, "r") as f:
        json_dict = json.load(f)
    q = []
    for kp_dict in json_dict["keypoints"]:
        q.append(_imgaug.augmentables.kps.Keypoint(x=kp_dict["x"] - o, y=kp_dict["y"] - p))
    if extract is not None:
        r = (b.height, b.width, 3)
    else:
        r = (643, 960, 3)
    s = _imgaug.augmentables.kps.KeypointsOnImage(q, shape=r)
    if size is not None:
        t = _b(r, size)
        s = s.on(t)
    return s


def quokka_bounding_boxes(size=None, extract=None):
    # Processes data.
    u, v = 0, 0
    if extract is not None:
        b = _a(extract)
        u = b.x1
        v = b.y1
    with open(_F3, "r") as f:
        json_dict = json.load(f)
    w = []
    for bb_dict in json_dict["bounding_boxes"]:
        w.append(
            _imgaug.augmentables.bbs.BoundingBox(
                x1=bb_dict["x1"] - u,
                y1=bb_dict["y1"] - v,
                x2=bb_dict["x2"] - u,
                y2=bb_dict["y2"] - v
            )
        )
    if extract is not None:
        x = (b.height, b.width, 3)
    else:
        x = (643, 960, 3)
    y = _imgaug.augmentables.bbs.BoundingBoxesOnImage(w, shape=x)
    if size is not None:
        z = _b(x, size)
        y = y.on(z)
    return y


def quokka_polygons(size=None, extract=None):
    # Processes data.
    a, b = 0, 0
    if extract is not None:
        bb_extract = _a(extract)
        a = bb_extract.x1
        b = bb_extract.y1
    with open(_F3, "r") as f:
        json_dict = json.load(f)
    c = []
    for poly_json in json_dict["polygons"]:
        c.append(
            _imgaug.augmentables.polys.Polygon([(point["x"] - a, point["y"] - b)
                                                for point in poly_json["keypoints"]])
        )
    if extract is not None:
        d = (bb_extract.height, bb_extract.width, 3)
    else:
        d = (643, 960, 3)
    e = _imgaug.augmentables.polys.PolygonsOnImage(c, shape=d)
    if size is not None:
        f = _b(d, size)
        e = e.on(f)
    return e