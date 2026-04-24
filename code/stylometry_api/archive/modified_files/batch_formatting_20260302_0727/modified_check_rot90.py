from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    transformations = [
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

    inputImage = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    keypoints = ia.quokka_keypoints(0.25)
    for name, augmentation in transformations:
        print(name, "...")
        deterministicAug = augmentation.to_deterministic()
        augmentedImages = deterministicAug.augment_images([inputImage] * 16)
        augmentedKeypoints = deterministicAug.augment_keypoints([keypoints] * 16)
        augmentedImages = [augmentedKeypoints_i.draw_on_image(augmentedImages_i, size=5)
                      for augmentedImages_i, augmentedKeypoints_i in zip(augmentedImages, augmentedKeypoints)]
        ia.imshow(ia.draw_grid(augmentedImages))

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmaps = ia.quokka_heatmap(0.10)
    for name, augmentation in transformations:
        print(name, "...")
        deterministicAug = augmentation.to_deterministic()
        augmentedImages = deterministicAug.augment_images([inputImage] * 16)
        augmentedHeatmaps = deterministicAug.augment_heatmaps([heatmaps] * 16)
        augmentedImages = [augmentedHeatmaps_i.draw_on_image(augmentedImages_i)[0]
                      for augmentedImages_i, augmentedHeatmaps_i in zip(augmentedImages, augmentedHeatmaps)]
        ia.imshow(ia.draw_grid(augmentedImages))


if __name__ == "__main__":
    main()