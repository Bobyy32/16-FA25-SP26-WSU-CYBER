from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    z = [
        ("iaa.Rot90(-1, keep_size=False)", iaa.Rot90(-1, keep_size=False)),
        ("iaa.Rot90(0, keep_size=False)", iaa.Rot90(0, keep_size=False)),
        ("iaa.Rot90(1, keep_size=False)", iaa.Rot90(1, keep_size=False)),
        ("iaa.Rot90(2, keep_size=False)", iaa.Rot90(2, keep_size=False)),
        ("iaa.Rot90(3, keep_size=False)", iaa.Rot90(3, keep_size=False)),
        ("iaa.Rot90(4, keep_size=False)", iaa.Rot90(4, keep_size=False)),
        ("iaa.Rot90(-1, keep_size=True)", iaa.Rot90(-1, keep_size=True)),
        ("iaa.Rot90(0, keep_size=True)", iaa.Rot90(0, keep_size=True)),
        ("iaa.Rot90(1, keep_size=True)", iaa.Rot90(1, keep_size=True)),
        ("iaa.Rot90(2, keep_size=True)", iaa.Rot90(2, keep_size=True)),
        ("iaa.Rot90(3, keep_size=True)", iaa.Rot90(3, keep_size=True)),
        ("iaa.Rot90(4, keep_size=True)", iaa.Rot90(4, keep_size=True)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=False)", iaa.Rot90([0, 1, 2, 3, 4], keep_size=False)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=True)", iaa.Rot90([0, 1, 2, 3, 4], keep_size=True)),
        ("iaa.Rot90((0, 4), keep_size=False)", iaa.Rot90((0, 4), keep_size=False)),
        ("iaa.Rot90((0, 4), keep_size=True)", iaa.Rot90((0, 4), keep_size=True)),
        ("iaa.Rot90((1, 3), keep_size=False)", iaa.Rot90((1, 3), keep_size=False)),
        ("iaa.Rot90((1, 3), keep_size=True)", iaa.Rot90((1, 3), keep_size=True))
    ]

    q = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    k = ia.quokka_keypoints(0.25)
    for o, p in z:
        print(o, "...")
        d = p.to_deterministic()
        m = d.augment_images([q] * 16)
        b = d.augment_keypoints([k] * 16)
        m = [b_i.draw_on_image(m_i, size=5)
             for m_i, b_i in zip(m, b)]
        ia.imshow(ia.draw_grid(m))

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    h = ia.quokka_heatmap(0.10)
    for o, p in z:
        print(o, "...")
        d = p.to_deterministic()
        m = d.augment_images([q] * 16)
        b = d.augment_heatmaps([h] * 16)
        m = [b_i.draw_on_image(m_i)[0]
             for m_i, b_i in zip(m, b)]
        ia.imshow(ia.draw_grid(m))


if __name__ == "__main__":
    main()