from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def processAugmentations():
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

    sampleImage = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    keypoints = ia.quokka_keypoints(0.25)
    
    iterationIndex = 0
    while iterationIndex < len(transformations):
        name, augmentation = transformations[iterationIndex]
        print(name, "...")
        deterministicAug = augmentation.to_deterministic()
        augmentedImages = deterministicAug.augment_images([sampleImage] * 16)
        augmentedKeypoints = deterministicAug.augment_keypoints([keypoints] * 16)
        augmentedImages = [augmentedKeypoints_i.draw_on_image(augmentedImages_i, size=5)
                          for augmentedImages_i, augmentedKeypoints_i in zip(augmentedImages, augmentedKeypoints)]
        ia.imshow(ia.draw_grid(augmentedImages))
        iterationIndex += 1

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmaps = ia.quokka_heatmap(0.10)
    
    index = 0
    while index < len(transformations):
        name, aug = transformations[index]
        print(name, "...")
        augDet = aug.to_deterministic()
        imgAug = augDet.augment_images([sampleImage] * 16)
        heatmapsAug = augDet.augment_heatmaps([heatmaps] * 16)
        imgAug = [heatmapsAug_i.draw_on_image(imgAug_i)[0]
                 for imgAug_i, heatmapsAug_i in zip(imgAug, heatmapsAug)]
        ia.imshow(ia.draw_grid(imgAug))
        index += 1


if __name__ == "__main__":
    processAugmentations()