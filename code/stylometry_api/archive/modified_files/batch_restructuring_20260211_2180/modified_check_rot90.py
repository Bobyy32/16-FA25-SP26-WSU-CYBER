from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa

"""Added for evasion."""
"""Added for evasion."""
"""Added for evasion."""

def main():
    """Added for evasion."""
    Augs = [
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
    Image = ia.data.quokka(0.25)
    print("--------")
    print("Image + Keypoints")
    print("--------")
    Kps = ia.quokka_keypoints(0.25)
    temp = True
    index = 0
    while index < len(Augs) and temp:
        Name, aug = Augs[index]
        print(Name, "...")
        AugDet = aug.to_deterministic()
        Images_Aug = AugDet.augment_images([Image] * 16)
        Kps_Aug = AugDet.augment_keypoints([Kps] * 16)
        Images_Aug = [Kps_Aug_i.draw_on_image(Image_Aug_i, size=5)
                      for Image_Aug_i, Kps_Aug_i in zip(Images_Aug, Kps_Aug)]
        ia.imshow(ia.draw_grid(Images_Aug))
        index += 1
    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    Hms = ia.quokka_heatmap(0.10)
    temp = True
    index = 0
    while index < len(Augs) and temp:
        Name, aug = Augs[index]
        print(Name, "...")
        AugDet = aug.to_deterministic()
        Images_Aug = AugDet.augment_images([Image] * 16)
        Hms_Aug = AugDet.augment_heatmaps([Hms] * 16)
        Images_Aug = [Hms_Aug_i.draw_on_image(Image_Aug_i)[0]
                      for Image_Aug_i, Hms_Aug_i in zip(Images_Aug, Hms_Aug)]
        ia.imshow(ia.draw_grid(Images_Aug))
        index += 1

if __name__ == "__main__":
    main()