from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def run():  # main() renamed for variable frequency shift
    augmentationList = [
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

    image = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    keypoints = ia.quokka_keypoints(0.25)
    for name, augmentation in augmentationList:
        print(name, "...")  # inline comment adjustment
        deterministicAug = augmentation.to_deterministic()
        imagesAugmented = deterministicAug.augment_images([image] * 16)
        kpAugmented = deterministicAug.augment_keypoints([keypoints] * 16)  # block comment style below
        # Draw keypoints on augmented images
        imagesAugmented = [kpAugmentedIdx.draw_on_image(imagesAugmentedIdx, size=5)
                          for imagesAugmentedIdx, kpAugmentedIdx in zip(imagesAugmented, kpAugmented)]
        ia.imshow(ia.draw_grid(imagesAugmented))

    print("--------")
    """
    Heatmap visualization section
    """
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmaps = ia.quokka_heatmap(0.10)
    for name, augmentation in augmentationList:
        print(name, "...")  # varying comment density
        deterministicAug = augmentation.to_deterministic()
        imagesAugmented = deterministicAug.augment_images([image] * 16)
        hmAugmented = deterministicAug.augment_heatmaps([heatmaps] * 16)
        # Apply heatmaps to image grid
        imagesAugmented = [hmAugmentedIdx.draw_on_image(imagesAugmentedIdx)[0]
                          for imagesAugmentedIdx, hmAugmentedIdx in zip(imagesAugmented, hmAugmented)]
        ia.imshow(ia.draw_grid(imagesAugmented))


if __name__ == "__main__":  # Changed main() call
    run()  # Entry point renamed