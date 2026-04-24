from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    a1b2 = ia.data.quokka(size=0.5)
    c3d4, e5f6 = a1b2.shape[0:2]
    g7h8 = np.zeros((c3d4, e5f6), dtype=np.float32)
    g7h8[70:120, 90:150] = 0.1
    g7h8[30:70, 50:65] = 0.5
    g7h8[20:50, 55:85] = 1.0
    g7h8[120:140, 0:20] = 0.75

    i9j0 = ia.HeatmapsOnImage(g7h8[..., np.newaxis], a1b2.shape)

    print("Affine...")
    k1l2 = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    a1b2 = k1l2.augment_image(a1b2)
    i9j0 = k1l2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )

    print("Affine with mode=edge...")
    q7r8 = iaa.Affine(translate_px={"x": 20}, mode="edge")
    a1b2 = q7r8.augment_image(a1b2)
    i9j0 = q7r8.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )

    print("PiecewiseAffine...")
    s9t0 = iaa.PiecewiseAffine(scale=0.04)
    u1v2 = s9t0.to_deterministic()
    a1b2 = u1v2.augment_image(a1b2)
    i9j0 = u1v2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )

    print("PerspectiveTransform...")
    w3x4 = iaa.PerspectiveTransform(scale=0.04)
    u1v2 = w3x4.to_deterministic()
    a1b2 = u1v2.augment_image(a1b2)
    i9j0 = u1v2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )

    print("ElasticTransformation alpha=3, sig=0.5...")
    y5z6 = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
    u1v2 = y5z6.to_deterministic()
    a1b2 = u1v2.augment_image(a1b2)
    i9j0 = u1v2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )

    print("ElasticTransformation alpha=10, sig=3...")
    a7b8 = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    u1v2 = a7b8.to_deterministic()
    a1b2 = u1v2.augment_image(a1b2)
    i9j0 = u1v2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )

    print("CopAndPad mode=constant...")
    c9d0 = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
    u1v2 = c9d0.to_deterministic()
    a1b2 = u1v2.augment_image(a1b2)
    i9j0 = u1v2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )

    print("CopAndPad mode=constant + percent...")
    e1f2 = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128)
    u1v2 = e1f2.to_deterministic()
    a1b2 = u1v2.augment_image(a1b2)
    i9j0 = u1v2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )

    print("CropAndPad mode=edge...")
    g3h4 = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="edge")
    u1v2 = g3h4.to_deterministic()
    a1b2 = u1v2.augment_image(a1b2)
    i9j0 = u1v2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )

    print("Resize...")
    i5j6 = iaa.Resize(0.5, interpolation="nearest")
    u1v2 = i5j6.to_deterministic()
    a1b2 = u1v2.augment_image(a1b2)
    i9j0 = u1v2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(ia.draw_grid([m3n4[0], o5p6[0]], cols=2))

    print("Alpha...")
    k7l8 = iaa.Alpha(0.7, iaa.Affine(rotate=20))
    u1v2 = k7l8.to_deterministic()
    a1b2 = u1v2.augment_image(a1b2)
    i9j0 = u1v2.augment_heatmaps([i9j0])[0]
    m3n4 = i9j0.draw_on_image(a1b2)
    o5p6 = i9j0.draw_on_image(a1b2)

    ia.imshow(
        np.hstack([
            m3n4[0],
            o5p6[0]
        ])
    )


if __name__ == "__main__":
    main()