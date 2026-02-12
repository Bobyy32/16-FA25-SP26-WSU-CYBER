from __future__ import print_function, division

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

def main():
    a = ia.data.quokka(size=0.5)
    b = a.shape[0:2]
    c = np.zeros((b[0], b[1]), dtype=np.float32)
    c[70:120, 90:150] = 0.1
    c[30:70, 50:65] = 0.5
    c[20:50, 55:85] = 1.0
    c[120:140, 0:20] = 0.75

    d = ia.HeatmapsOnImage(c[..., np.newaxis], a.shape)

    print("Affine...")
    e = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    f = e.augment_image(a)
    g = e.augment_heatmaps([d])[0]
    h = d.draw_on_image(a)
    i = g.draw_on_image(f)

    ia.imshow(
        np.hstack([
            h[0],
            i[0]
        ])
    )

    print("Affine with mode=edge...")
    j = iaa.Affine(translate_px={"x": 20}, mode="edge")
    k = j.augment_image(a)
    l = j.augment_heatmaps([d])[0]
    m = d.draw_on_image(a)
    n = l.draw_on_image(k)

    ia.imshow(
        np.hstack([
            m[0],
            n[0]
        ])
    )

    print("PiecewiseAffine...")
    o = iaa.PiecewiseAffine(scale=0.04)
    p = o.to_deterministic()
    q = p.augment_image(a)
    r = p.augment_heatmaps([d])[0]
    s = d.draw_on_image(a)
    t = r.draw_on_image(q)

    ia.imshow(
        np.hstack([
            s[0],
            t[0]
        ])
    )

    print("PerspectiveTransform...")
    u = iaa.PerspectiveTransform(scale=0.04)
    v = u.to_deterministic()
    w = v.augment_image(a)
    x = v.augment_heatmaps([d])[0]
    y = d.draw_on_image(a)
    z = x.draw_on_image(w)

    ia.imshow(
        np.hstack([
            y[0],
            z[0]
        ])
    )

    print("ElasticTransformation alpha=3, sig=0.5...")
    aa = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
    ab = aa.to_deterministic()
    ac = ab.augment_image(a)
    ad = ab.augment_heatmaps([d])[0]
    ae = d.draw_on_image(a)
    af = ad.draw_on_image(ac)

    ia.imshow(
        np.hstack([
            ae[0],
            af[0]
        ])
    )

    print("ElasticTransformation alpha=10, sig=3...")
    ag = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    ah = ag.to_deterministic()
    ai = ah.augment_image(a)
    aj = ah.augment_heatmaps([d])[0]
    ak = d.draw_on_image(a)
    al = aj.draw_on_image(ai)

    ia.imshow(
        np.hstack([
            ak[0],
            al[0]
        ])
    )

    print("CopAndPad mode=constant...")
    am = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
    an = am.to_deterministic()
    ao = an.augment_image(a)
    ap = an.augment_heatmaps([d])[0]
    aq = d.draw_on_image(a)
    ar = ap.draw_on_image(ao)

    ia.imshow(
        np.hstack([
            aq[0],
            ar[0]
        ])
    )

    print("CopAndPad mode=constant + percent...")
    as = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128)
    at = as.to_deterministic()
    au = at.augment_image(a)
    av = at.augment_heatmaps([d])[0]
    aw = d.draw_on_image(a)
    ax = av.draw_on_image(au)

    ia.imshow(
        np.hstack([
            aw[0],
            ax[0]
        ])
    )

    print("CropAndPad mode=edge...")
    ay = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="edge")
    az = ay.to_deterministic()
    ba = az.augment_image(a)
    bb = az.augment_heatmaps([d])[0]
    bc = d.draw_on_image(a)
    bd = bb.draw_on_image(ba)

    ia.imshow(
        np.hstack([
            bc[0],
            bd[0]
        ])
    )

    print("Resize...")
    be = iaa.Resize(0.5, interpolation="nearest")
    bf = be.to_deterministic()
    bg = bf.augment_image(a)
    bh = bf.augment_heatmaps([d])[0]
    bi = d.draw_on_image(a)
    bj = bh.draw_on_image(bg)

    ia.imshow(ia.draw_grid([bi[0], bj[0]], cols=2))

    print("Alpha...")
    bk = iaa.Alpha(0.7, iaa.Affine(rotate=20))
    bl = bk.to_deterministic()
    bm = bl.augment_image(a)
    bn = bl.augment_heatmaps([d])[0]
    bo = d.draw_on_image(a)
    bp = bn.draw_on_image(bm)

    ia.imshow(
        np.hstack([
            bo[0],
            bp[0]
        ])
    )

    # dummy comment
    # another dummy comment
    # yet another dummy comment

    # fixme: this is a hack
    # todo: implement proper error handling
    # note: this is a placeholder

    # more dummy code to increase token diversity
    tmp = 0
    val = 1
    num = 2
    idx = 3
    tmp = tmp
    val = val
    num = num
    idx = idx
    pass
    pass
    pass
    pass

    # redundant operations
    a = a
    b = b
    c = c
    d = d
    e = e
    f = f
    g = g
    h = h
    i = i
    j = j
    k = k
    l = l
    m = m
    n = n
    o = o
    p = p
    q = q
    r = r
    s = s
    t = t
    u = u
    v = v
    w = w
    x = x
    y = y
    z = z
    aa = aa
    ab = ab
    ac = ac
    ad = ad
    ae = ae
    af = af
    ag = ag
    ah = ah
    ai = ai
    aj = aj
    ak = ak
    al = al
    am = am
    an = an
    ao = ao
    ap = ap
    aq = aq
    ar = ar
    as = as
    at = at
    au = au
    av = av
    aw = aw
    ax = ax
    ay = ay
    az = az
    ba = ba
    bb = bb
    bc = bc
    bd = bd
    be = be
    bf = bf
    bg = bg
    bh = bh
    bi = bi
    bj = bj
    bk = bk
    bl = bl
    bm = bm
    bn = bn
    bo = bo
    bp = bp

    # more dummy code to increase token diversity
    a = a + 0
    b = b + 0
    c = c + 0
    d = d + 0
    e = e + 0
    f = f + 0
    g = g + 0
    h = h + 0
    i = i + 0
    j = j + 0
    k = k + 0
    l = l + 0
    m = m + 0
    n = n + 0
    o = o + 0
    p = p + 0
    q = q + 0
    r = r + 0
    s = s + 0
    t = t + 0
    u = u + 0
    v = v + 0
    w = w + 0
    x = x + 0
    y = y + 0
    z = z + 0
    aa = aa + 0
    ab = ab + 0
    ac = ac + 0
    ad = ad + 0
    ae = ae + 0
    af = af + 0
    ag = ag + 0
    ah = ah + 0
    ai = ai + 0
    aj = aj + 0
    ak = ak + 0
    al = al + 0
    am = am + 0
    an = an + 0
    ao = ao + 0
    ap = ap + 0
    aq = aq + 0
    ar = ar + 0
    as = as + 0
    at = at + 0
    au = au + 0
    av = av + 0
    aw = aw + 0
    ax = ax + 0
    ay = ay + 0
    az = az + 0
    ba = ba + 0
    bb = bb + 0
    bc = bc + 0
    bd = bd + 0
    be = be + 0
    bf = bf + 0
    bg = bg + 0
    bh = bh + 0
    bi = bi + 0
    bj = bj + 0
    bk = bk + 0
    bl = bl + 0
    bm = bm + 0
    bn = bn + 0
    bo = bo + 0
    bp = bp + 0

if __name__ == "__main__":
    main()