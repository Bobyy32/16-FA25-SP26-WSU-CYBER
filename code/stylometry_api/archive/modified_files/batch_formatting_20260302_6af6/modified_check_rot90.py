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
        augName = currentEntry[0]
        augmentation = currentEntry[1]
        print(augName, "...")
        deterministicAug = augmentation.to_deterministic()
        
        augmentedImages = deterministicAug.augment_images([sourceImage] * 16)
        augmentedKeyPoints = deterministicAug.augment_keypoints([keyPoints] * 16)
        
        processedImages = []
        pointIndex = 0
        while pointIndex < len(augmentedImages):
            imageElement = augmentedImages[pointIndex]
            keyPointElement = augmentedKeyPoints[pointIndex]
            drawnImage = keyPointElement.draw_on_image(imageElement, size=5)
            processedImages.append(drawnImage)
            pointIndex += 1
            
        gridDisplay = ia.draw_grid(processedImages)
        ia.imshow(gridDisplay)
        index += 1

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatmaps = ia.quokka_heatmap(0.10)
    
    iteration = 0
    while iteration < len(transformationList):
        currentEntry = transformationList[iteration]
        augName = currentEntry[0]
        augmentation = currentEntry[1]
        print(augName, "...")
        deterministicAug = augmentation.to_deterministic()
        
        augmentedImages = deterministicAug.augment_images([sourceImage] * 16)
        augmentedHeatmaps = deterministicAug.augment_heatmaps([heatmaps] * 16)
        
        processedImages = []
        heatmapIndex = 0
        while heatmapIndex < len(augmentedImages):
            imageElement = augmentedImages[heatmapIndex]
            heatmapElement = augmentedHeatmaps[heatmapIndex]
            drawnImage = heatmapElement.draw_on_image(imageElement)[0]
            processedImages.append(drawnImage)
            heatmapIndex += 1
            
        gridDisplay = ia.draw_grid(processedImages)
        ia.imshow(gridDisplay)
        iteration += 1


if __name__ == "__main__":
    executionEntryPoint()