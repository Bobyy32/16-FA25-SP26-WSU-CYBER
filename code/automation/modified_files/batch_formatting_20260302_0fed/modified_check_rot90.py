from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def executionEntryPoint():
    transformationList = [
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

    sourceImage = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    keyPoints = ia.quokka_keypoints(0.25)
    
    index = 0
    while index < len(transformationList):
        currentEntry = transformationList[index]
        name = currentEntry[0]
        augmentation = currentEntry[1]
        print(name, "...")
        deterministicAug = augmentation.to_deterministic()
        
        batchImages = [sourceImage] * 16
        augmentedImages = deterministicAug.augment_images(batchImages)
        batchKeyPoints = [keyPoints] * 16
        augmentedKeyPoints = deterministicAug.augment_keypoints(batchKeyPoints)
        
        finalImages = []
        pointIndex = 0
        while pointIndex < len(augmentedImages):
            imageElement = augmentedImages[pointIndex]
            keyPointElement = augmentedKeyPoints[pointIndex]
            drawnImage = keyPointElement.draw_on_image(imageElement, size=5)
            finalImages.append(drawnImage)
            pointIndex += 1
            
        gridDisplay = ia.draw_grid(finalImages)
        ia.imshow(gridDisplay)
        index += 1

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmaps = ia.quokka_heatmap(0.10)
    
    counter = 0
    while counter < len(transformationList):
        entry = transformationList[counter]
        name = entry[0]
        aug = entry[1]
        print(name, "...")
        deterministicAug = aug.to_deterministic()
        
        batchImages = [sourceImage] * 16
        augmentedImages = deterministicAug.augment_images(batchImages)
        batchHeatmaps = [heatmaps] * 16
        augmentedHeatmaps = deterministicAug.augment_heatmaps(batchHeatmaps)
        
        resultImages = []
        imageIndex = 0
        while imageIndex < len(augmentedImages):
            imageElement = augmentedImages[imageIndex]
            heatmapElement = augmentedHeatmaps[imageIndex]
            drawnImage = heatmapElement.draw_on_image(imageElement)[0]
            resultImages.append(drawnImage)
            imageIndex += 1
            
        gridDisplay = ia.draw_grid(resultImages)
        ia.imshow(gridDisplay)
        counter += 1


if __name__ == "__main__":
    executionEntryPoint()