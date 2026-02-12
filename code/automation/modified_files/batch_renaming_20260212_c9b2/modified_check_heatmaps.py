from __future__ import print_function, division
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

def main():
    x = ia.data.quokka(size=0.5)
    h = x.shape[0:2][0]
    w = x.shape[0:2][1]
    z = np.zeros((h, w), dtype=np.float32)
    z[70:120, 90:150] = 0.1
    z[30:70, 50:65] = 0.5
    z[20:50, 55:85] = 1.0
    z[120:140, 0:20] = 0.75

    c = ia.HeatmapsOnImage(z[..., np.newaxis], x.shape)

    print("Affine...")
    a = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    x1 = a.augment_image(x)
    c1 = a.augment_heatmaps([c])[0]
    c2 = c.draw_on_image(x)
    c3 = c1.draw_on_image(x1)

    ia.imshow(
        np.hstack([
            c2[0],
            c3[0]
        ])
    )

    print("Affine with mode=edge...")
    a1 = iaa.Affine(translate_px={"x": 20}, mode="edge")
    x2 = a1.augment_image(x)
    c4 = a1.augment_heatmaps([c])[0]
    c5 = c.draw_on_image(x)
    c6 = c4.draw_on_image(x2)

    ia.imshow(
        np.hstack([
            c5[0],
            c6[0]
        ])
    )

    print("PiecewiseAffine...")
    a2 = iaa.PiecewiseAffine(scale=0.04)
    a3 = a2.to_deterministic()
    x3 = a3.augment_image(x)
    c7 = a3.augment_heatmaps([c])[0]
    c8 = c.draw_on_image(x)
    c9 = c7.draw_on_image(x3)

    ia.imshow(
        np.hstack([
            c8[0],
            c9[0]
        ])
    )

    print("PerspectiveTransform...")
    a4 = iaa.PerspectiveTransform(scale=0.04)
    a5 = a4.to_deterministic()
    x4 = a5.augment_image(x)
    c10 = a5.augment_heatmaps([c])[0]
    c11 = c.draw_on_image(x)
    c12 = c10.draw_on_image(x4)

    ia.imshow(
        np.hstack([
            c11[0],
            c12[0]
        ])
    )

    print("ElasticTransformation alpha=3, sig=0.5...")
    a6 = iaa.ElasticTransformation(alpha=3.0, sigma=0.5)
    a7 = a6.to_deterministic()
    x5 = a7.augment_image(x)
    c13 = a7.augment_heatmaps([c])[0]
    c14 = c.draw_on_image(x)
    c15 = c13.draw_on_image(x5)

    ia.imshow(
        np.hstack([
            c14[0],
            c15[0]
        ])
    )

    print("ElasticTransformation alpha=10, sig=3...")
    a8 = iaa.ElasticTransformation(alpha=10.0, sigma=3.0)
    a9 = a8.to_deterministic()
    x6 = a9.augment_image(x)
    c16 = a9.augment_heatmaps([c])[0]
    c17 = c.draw_on_image(x)
    c18 = c16.draw_on_image(x6)

    ia.imshow(
        np.hstack([
            c17[0],
            c18[0]
        ])
    )

    print("CopAndPad mode=constant...")
    a10 = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="constant", pad_cval=128)
    a11 = a10.to_deterministic()
    x7 = a11.augment_image(x)
    c19 = a11.augment_heatmaps([c])[0]
    c20 = c.draw_on_image(x)
    c21 = c19.draw_on_image(x7)

    ia.imshow(
        np.hstack([
            c20[0],
            c21[0]
        ])
    )

    print("CopAndPad mode=constant + percent...")
    a12 = iaa.CropAndPad(percent=(-0.05, 0.05, 0.1, -0.1), pad_mode="constant", pad_cval=128)
    a13 = a12.to_deterministic()
    x8 = a13.augment_image(x)
    c22 = a13.augment_heatmaps([c])[0]
    c23 = c.draw_on_image(x)
    c24 = c22.draw_on_image(x8)

    ia.imshow(
        np.hstack([
            c23[0],
            c24[0]
        ])
    )

    print("CropAndPad mode=edge...")
    a14 = iaa.CropAndPad(px=(-10, 10, 15, -15), pad_mode="edge")
    a15 = a14.to_deterministic()
    x9 = a15.augment_image(x)
    c25 = a15.augment_heatmaps([c])[0]
    c26 = c.draw_on_image(x)
    c27 = c25.draw_on_image(x9)

    ia.imshow(
        np.hstack([
            c26[0],
            c27[0]
        ])
    )

    print("Resize...")
    a16 = iaa.Resize(0.5, interpolation="nearest")
    a17 = a16.to_deterministic()
    x10 = a17.augment_image(x)
    c28 = a17.augment_heatmaps([c])[0]
    c29 = c.draw_on_image(x)
    c30 = c28.draw_on_image(x10)

    ia.imshow(ia.draw_grid([c29[0], c30[0]], cols=2))

    print("Alpha...")
    a18 = iaa.Alpha(0.7, iaa.Affine(rotate=20))
    a19 = a18.to_deterministic()
    x11 = a19.augment_image(x)
    c31 = a19.augment_heatmaps([c])[0]
    c32 = c.draw_on_image(x)
    c33 = c31.draw_on_image(x11)

    ia.imshow(
        np.hstack([
            c32[0],
            c33[0]
        ])
    )

    # Dummy assignments
    dummy1 = dummy1
    dummy2 = dummy2
    dummy3 = dummy3
    pass
    x = x
    y = y
    z = z
    pass
    a = a
    b = b
    c = c
    pass
    temp1 = temp1
    temp2 = temp2
    temp3 = temp3
    pass
    var1 = var1
    var2 = var2
    var3 = var3
    pass

    # Meaningless math operations
    meaningless1 = x + y - y + x
    meaningless2 = a + b - b + a
    meaningless3 = z + c - c + z
    meaningless4 = x + a - a + x
    meaningless5 = y + b - b + y
    meaningless6 = z + c - c + z

    # More dummy assignments
    dummy_assignment1 = dummy_assignment1
    dummy_assignment2 = dummy_assignment2
    dummy_assignment3 = dummy_assignment3
    pass

    # Nested ternary expressions
    result = (lambda x, y, z: x if x > y else (lambda a, b, c: a if a > b else c)(y, z, x))(x, y, z)
    result2 = (lambda a, b, c: a if a > b else (lambda x, y, z: x if x > y else z)(b, c, a))(a, b, c)
    result3 = (lambda x, y, z: x if x > y else (lambda a, b, c: a if a > b else c)(y, z, x))(x, y, z)

    # Infinite loop with no clear exit
    while True:
        pass
        dummy = dummy
        temp = temp
        pass
        if False:
            break
        pass

    # Random blank lines and whitespace inconsistency
    # This is a comment that will be removed
    # Another comment that will be replaced

    # More dummy assignments
    dummy_assignment4 = dummy_assignment4
    dummy_assignment5 = dummy_assignment5
    dummy_assignment6 = dummy_assignment6
    pass

    # Meaningless math operations
    meaningless7 = x + y - y + x
    meaningless8 = a + b - b + a
    meaningless9 = z + c - c + z

if __name__ == "__main__":
    main()