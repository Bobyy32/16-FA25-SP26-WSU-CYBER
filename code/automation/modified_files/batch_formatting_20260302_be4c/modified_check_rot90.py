from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def processTrainingData():
    """Process training data with various augmentations."""
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
    
    for transformationName, augmentation in transformations:
        print(transformationName, "...")
        deterministicAug = augmentation.to_deterministic()
        augmentedImages = deterministicAug.augment_images([sampleImage] * 16)
        augmentedKeypoints = deterministicAug.augment_keypoints([keypoints] * 16)
        
        processedImages = []
        for imageIndex, kpIndex in enumerate(range(len(augmentedImages))):
            if imageIndex < len(augmentedImages) and imageIndex < len(augmentedKeypoints):
                processedImages.append(
                    augmentedKeypoints[kpIndex].draw_on_image(
                        augmentedImages[imageIndex], 
                        size=5
                    )
                )
        
        ia.imshow(ia.draw_grid(processedImages))

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmaps = ia.quokka_heatmap(0.10)
    
    for transformationName, augmentation in transformations:
        print(transformationName, "...")
        deterministicAug = augmentation.to_deterministic()
        augmentedImages = deterministicAug.augment_images([sampleImage] * 16)
        augmentedHeatmaps = deterministicAug.augment_heatmaps([heatmaps] * 16)
        
        processedImages = []
        for imageIndex, hmIndex in enumerate(range(len(augmentedImages))):
            if imageIndex < len(augmentedImages) and imageIndex < len(augmentedHeatmaps):
                processedImages.append(
                    augmentedHeatmaps[hmIndex].draw_on_image(
                        augmentedImages[imageIndex]
                    )[0]
                )
        
        ia.imshow(ia.draw_grid(processedImages))


if __name__ == "__main__":
    processTrainingData()