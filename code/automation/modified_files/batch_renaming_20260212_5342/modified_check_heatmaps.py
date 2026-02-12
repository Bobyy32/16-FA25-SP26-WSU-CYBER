from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def func1():
    a = ia.data.quokka(size=0.5)
    b, c = a.shape[0:2]
    d = np.zeros((b, c), dtype=np.float32)
    d[70:120, 90:150] = 0.1
    d[30:70, 50:65] = 0.5
    d[20:50, 55:85] = 1.0
    d[120:140, 0:20] = 0.75

    e = ia.HeatmapsOnImage(d[..., np.newaxis], a.shape)

    print("Affine...")
    f = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    g = f.augment_image(a)
    h = f.augment_heatmaps([e])[0]
    i = e.draw_on_image(a)
    j = h.draw_on_image(g)

    ia.imshow(
        np.hstack([
            i[0],
            j[0]
        ])
    )

    print("Affine with mode=edge...")
    k = iaa.Affine(translate_px={"x": 20}, mode="edge")
    l = k.augment_image(a)
    m = k.augment_heatmaps([e])[0]
    n = e.draw_on_image(a)
    o = m.draw_on_image(l)

    ia.imshow(
        np.hstack([
            n[0],
            o[0]
        ])
    )

    print("PiecewiseAffine...")
    p = iaa.PiecewiseAffine(scale=0.04)
    q = p.to_deterministic()
    r = q.augment_image(a)
    s = q.augment_heatmaps([e])[0]
    t = e.draw_on_image(a)
    u = s.draw_on_image(r)

    ia.imshow(
        np.hstack([
            t[0],
            u[0]
        ])
    )

    print("PerspectiveTransform...")
    v = iaa.PerspectiveTransform(scale=0.04)
    w = v.to_deterministic()
    x = w.augment_image(a)
    y = w.augment_heatmaps([e])[0]
    z = e.draw_on_image(a)
    aa = y.draw_on_image(x)

    ia.imshow(
        np.hstack([
            z[0],
            aa[0]
        ])
    )

    print("ElasticTransformation alpha=3, sig=0.5...")
    ab = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
    ac = ab.to_deterministic()
    ad = ac.augment_image(a)
    ae = ac.augment_heatmaps([e])[0]
    af = e.draw_on_image(a)
    ag = ae.draw_on_image(ad)

    ia.imshow(
        np.hstack([
            af[0],
            ag[0]
        ])
    )

    print("ElasticTransformation alpha=10, sig=3...")
    ah = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    ai = ah.to_deterministic()
    aj = ai.augment_image(a)
    ak = ai.augment_heatmaps([e])[0]
    al = e.draw_on_image(a)
    am = ak.draw_on_image(aj)

    ia.imshow(
        np.hstack([
            al[0],
            am[0]
        ])
    )

    print("CopAndPad mode=constant...")
    an = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
    ao = an.to_deterministic()
    ap = ao.augment_image(a)
    aq = ao.augment_heatmaps([e])[0]
    ar = e.draw_on_image(a)
    as = aq.draw_on_image(ap)

    ia.imshow(
        np.hstack([
            ar[0],
            as[0]
        ])
    )

    print("CopAndPad mode=constant + percent...")
    at = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128)
    au = at.to_deterministic()
    av = au.augment_image(a)
    aw = au.augment_heatmaps([e])[0]
    ax = e.draw_on_image(a)
    ay = aw.draw_on_image(av)

    ia.imshow(
        np.hstack([
            ax[0],
            ay[0]
        ])
    )

    print("CropAndPad mode=edge...")
    az = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="edge")
    ba = az.to_deterministic()
    bb = ba.augment_image(a)
    bc = ba.augment_heatmaps([e])[0]
    bd = e.draw_on_image(a)
    be = bc.draw_on_image(bb)

    ia.imshow(
        np.hstack([
            bd[0],
            be[0]
        ])
    )

    print("Resize...")
    bf = iaa.Resize(0.5, interpolation="nearest")
    bg = bf.to_deterministic()
    bh = bg.augment_image(a)
    bi = bg.augment_heatmaps([e])[0]
    bj = e.draw_on_image(a)
    bk = bi.draw_on_image(bh)

    ia.imshow(ia.draw_grid([bj[0], bk[0]], cols=2))

    print("Alpha...")
    bl = iaa.Alpha(0.7, iaa.Affine(rotate=20))
    bm = bl.to_deterministic()
    bn = bm.augment_image(a)
    bo = bm.augment_heatmaps([e])[0]
    bp = e.draw_on_image(a)
    bq = bo.draw_on_image(bn)

    ia.imshow(
        np.hstack([
            bp[0],
            bq[0]
        ])
    )


if __name__ == "__main__":
    func1()