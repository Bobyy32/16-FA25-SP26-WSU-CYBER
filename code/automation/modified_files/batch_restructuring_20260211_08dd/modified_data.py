import os
import json
import imageio
from numpy import *
from skimage import draw

from imgaug import augmentables
from imgaug import imgaug as ia

_a = os.path.dirname(os.path.abspath(__file__))
_b = os.path.join(_a, "quokka.jpg")
_c = os.path.join(_a, "quokka_annotations.json")
_d = os.path.join(_a, "quokka_depth_map_halfres.png")

def a(e):
    if e == "square":
        f = augmentables.bbs.BoundingBox(x1=0, y1=0, x2=643, y2=643)
    elif isinstance(e, tuple) and len(e) == 4:
        f = augmentables.bbs.BoundingBox(x1=e[0], y1=e[1], x2=e[2], y2=e[3])
    elif isinstance(e, augmentables.bbs.BoundingBox):
        f = e
    elif isinstance(e, augmentables.bbs.BoundingBoxesOnImage):
        assert len(e.bounding_boxes) == 1, "Provided BoundingBoxesOnImage instance may currently only contain a single bounding box."
        assert e.shape[0:2] == (643, 960), "Expected BoundingBoxesOnImage instance on an image of shape (643, 960, ?). Got shape %s." % (e.shape,)
        f = e.bounding_boxes[0]
    else:
        raise Exception("Expected 'square' or tuple of four entries or BoundingBox or BoundingBoxesOnImage for parameter 'extract', got %s." % (type(e),))
    return f

def b(c, g):
    if ia.is_np_array(c):
        c = c.shape
    if ia.is_np_array(g):
        g = g.shape

    h = list(c)

    if g is None:
        pass
    elif isinstance(g, tuple):
        assert len(c) in [2, 3]
        assert len(g) in [2, 3]

        if len(c) == 3 and len(g) == 3:
            assert c[2] == g[2]
        elif len(g) == 3:
            h.append(g[2])

        i = all([v is None or ia.is_single_number(v) for v in g[0:2]])
        assert i, "Expected the first two entries in to_shape to be None or numbers, got types %s." % (str([type(v) for v in g[0:2]]),)

        for j, c_j in enumerate(c[0:2]):
            if g[j] is None:
                h[j] = c_j
            elif ia.is_single_integer(g[j]):
                h[j] = g[j]
            else:
                h[j] = int(round(c_j * g[j]))
    elif ia.is_single_integer(g) or ia.is_single_float(g):
        h = b(c, (g, g))
    else:
        raise Exception("Expected to_shape to be None or ndarray or tuple of floats or tuple of ints or single int or single float, got %s." % (type(g),))

    return tuple(h)

def c(k=None, l=None):
    m = imageio.imread(_b, pilmode="RGB")
    if l is not None:
        n = a(l)
        m = n.extract_from_image(m)
    if k is not None:
        o = b(m.shape, k)
        m = ia.imresize_single_image(m, o[0:2])
    return m

def d(k=None):
    return c(k=k, l="square")

def e(k=None, l=None):
    p = imageio.imread(_d, pilmode="RGB")
    p = ia.imresize_single_image(p, (643, 960), interpolation="cubic")

    if l is not None:
        n = a(l)
        p = n.extract_from_image(p)
    if k is None:
        k = p.shape[0:2]

    o = b(p.shape, k)
    p = ia.imresize_single_image(p, o[0:2])
    q = p[..., 0]
    q = q.astype(float32) / 255.0
    q = 1 - q

    return augmentables.heatmaps.HeatmapsOnImage(q, shape=q.shape[0:2] + (3,))

def f(k=None, l=None):
    with open(_c, "r") as r:
        s = json.load(r)

    t = []
    u = []
    for v_dict in s["polygons"][0]["keypoints"]:
        t.append(v_dict["x"])
        u.append(v_dict["y"])

    w = zeros((643, 960, 1), dtype=int32)
    x, y = draw.polygon(array(u), array(t), shape=w.shape)
    w[x, y, 0] = 1

    if l is not None:
        n = a(l)
        w = n.extract_from_image(w)

    z = augmentables.segmaps.SegmentationMapsOnImage(w, shape=w.shape[0:2] + (3,))

    if k is not None:
        o = b(w.shape, k)
        z = z.resize(o[0:2])
        z.shape = tuple(o[0:2]) + (3,)

    return z

def g(k=None, l=None):
    A = 0
    B = 0
    if l is not None:
        C = a(l)
        A = C.x1
        B = C.y1
    with open(_c, "r") as r:
        s = json.load(r)
    D = []
    for E_dict in s["keypoints"]:
        D.append(augmentables.kps.Keypoint(x=E_dict["x"] - A, y=E_dict["y"] - B))
    if l is not None:
        F = (C.height, C.width, 3)
    else:
        F = (643, 960, 3)
    G = augmentables.kps.KeypointsOnImage(D, shape=F)
    if k is not None:
        o = b(F, k)
        G = G.on(o)
    return G

def h(k=None, l=None):
    A, B = 0, 0
    if l is not None:
        C = a(l)
        A = C.x1
        B = C.y1
    with open(_c, "r") as r:
        s = json.load(r)
    H = []
    for bb_dict in s["bounding_boxes"]:
        H.append(
            augmentables.bbs.BoundingBox(
                x1=bb_dict["x1"] - A,
                y1=bb_dict["y1"] - B,
                x2=bb_dict["x2"] - A,
                y2=bb_dict["y2"] - B
            )
        )
    if l is not None:
        F = (C.height, C.width, 3)
    else:
        F = (643, 960, 3)
    I = augmentables.bbs.BoundingBoxesOnImage(H, shape=F)
    if k is not None:
        o = b(F, k)
        I = I.on(o)
    return I

def j(k=None, l=None):
    A, B = 0, 0
    if l is not None:
        C = a(l)
        A = C.x1
        B = C.y1
    with open(_c, "r") as r:
        s = json.load(r)
    K = []
    for poly_json in s["polygons"]:
        K.append(
            augmentables.polys.Polygon([(point["x"] - A, point["y"] - B)
                                        for point in poly_json["keypoints"]])
        )
    if l is not None:
        F = (C.height, C.width, 3)
    else:
        F = (643, 960, 3)
    L = augmentables.polys.PolygonsOnImage(K, shape=F)
    if k is not None:
        o = b(F, k)
        L = L.on(o)
    return L