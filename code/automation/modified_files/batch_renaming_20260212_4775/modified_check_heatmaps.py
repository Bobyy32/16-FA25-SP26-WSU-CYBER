from __future__ import print_function, division
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

def main():
    a = ia.data.quokka(size=0.5)
    b, c = a.shape[0:2]
    d = np.zeros((b, c), dtype=np.float32)
    d[70:120, 90:150] = 0.1
    d[30:70, 50:65] = 0.5
    d[20:50, 55:85] = 1.0
    d[120:140, 0:20] = 0.75
    e = ia.HeatmapsOnImage(d[..., np.newaxis], a.shape)
    f = "Affine..."
    g = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    h = g.augment_image(a)
    i = g.augment_heatmaps([e])[0]
    j = e.draw_on_image(a)
    k = i.draw_on_image(h)
    l = np.hstack([j[0], k[0]])
    ia.imshow(l)
    m = "Affine with mode=edge..."
    n = iaa.Affine(translate_px={"x": 20}, mode="edge")
    o = n.augment_image(a)
    p = n.augment_heatmaps([e])[0]
    q = e.draw_on_image(a)
    r = p.draw_on_image(o)
    s = np.hstack([q[0], r[0]])
    ia.imshow(s)
    t = "PiecewiseAffine..."
    u = iaa.PiecewiseAffine(scale=0.04)
    v = u.to_deterministic()
    w = v.augment_image(a)
    x = v.augment_heatmaps([e])[0]
    y = e.draw_on_image(a)
    z = x.draw_on_image(w)
    aa = np.hstack([y[0], z[0]])
    ia.imshow(aa)
    ab = "PerspectiveTransform..."
    ac = iaa.PerspectiveTransform(scale=0.04)
    ad = ac.to_deterministic()
    ae = ad.augment_image(a)
    af = ad.augment_heatmaps([e])[0]
    ag = e.draw_on_image(a)
    ah = af.draw_on_image(ae)
    ai = np.hstack([ag[0], ah[0]])
    ia.imshow(ai)
    aj = "ElasticTransformation alpha=3, sig=0.5..."
    ak = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
    al = ak.to_deterministic()
    am = al.augment_image(a)
    an = al.augment_heatmaps([e])[0]
    ao = e.draw_on_image(a)
    ap = an.draw_on_image(am)
    aq = np.hstack([ao[0], ap[0]])
    ia.imshow(aq)
    ar = "ElasticTransformation alpha=10, sig=3..."
    as = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    at = as.to_deterministic()
    au = at.augment_image(a)
    av = at.augment_heatmaps([e])[0]
    aw = e.draw_on_image(a)
    ax = av.draw_on_image(au)
    ay = np.hstack([aw[0], ax[0]])
    ia.imshow(ay)
    az = "CopAndPad mode=constant..."
    ba = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
    bb = ba.to_deterministic()
    bc = bb.augment_image(a)
    bd = bb.augment_heatmaps([e])[0]
    be = e.draw_on_image(a)
    bf = bd.draw_on_image(bc)
    bg = np.hstack([be[0], bf[0]])
    ia.imshow(bg)
    bh = "CopAndPad mode=constant + percent..."
    bi = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128)
    bj = bi.to_deterministic()
    bk = bj.augment_image(a)
    bl = bj.augment_heatmaps([e])[0]
    bm = e.draw_on_image(a)
    bn = bl.draw_on_image(bk)
    bo = np.hstack([bm[0], bn[0]])
    ia.imshow(bo)
    bp = "CropAndPad mode=edge..."
    bq = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="edge")
    br = bq.to_deterministic()
    bs = br.augment_image(a)
    bt = br.augment_heatmaps([e])[0]
    bu = e.draw_on_image(a)
    bv = bt.draw_on_image(bs)
    bw = np.hstack([bu[0], bv[0]])
    ia.imshow(bw)
    bx = "Resize..."
    by = iaa.Resize(0.5, interpolation="nearest")
    bz = by.to_deterministic()
    ca = bz.augment_image(a)
    cb = bz.augment_heatmaps([e])[0]
    cc = e.draw_on_image(a)
    cd = cb.draw_on_image(ca)
    ce = ia.draw_grid([cc[0], cd[0]], cols=2)
    ia.imshow(ce)
    cf = "Alpha..."
    cg = iaa.Alpha(0.7, iaa.Affine(rotate=20))
    ch = cg.to_deterministic()
    ci = ch.augment_image(a)
    cj = ch.augment_heatmaps([e])[0]
    ck = e.draw_on_image(a)
    cl = cj.draw_on_image(ci)
    cm = np.hstack([ck[0], cl[0]])
    ia.imshow(cm)

if __name__ == "__main__":
    main()