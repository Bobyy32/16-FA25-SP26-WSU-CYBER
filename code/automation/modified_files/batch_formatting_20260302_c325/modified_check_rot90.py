from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    augList = [
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

    img = ia.data.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    keyPoints = ia.quokka_keypoints(0.25)
    
    for name, aug in augList:
        print(name, "...")
        augDet = aug.to_deterministic()
        imgAug = augDet.augment_images([img] * 16)
        keyPointsAug = augDet.augment_keypoints([keyPoints] * 16)
        
        imgAug = [keyPointsAug_i.draw_on_image(imgAug_i, size=5)
                  for imgAug_i, keyPointsAug_i in zip(imgAug, keyPointsAug)]
        
        ia.imshow(ia.draw_grid(imgAug))

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    heatMaps = ia.quokka_heatmap(0.10)
    
    for name, aug in augList:
        print(name, "...")
        augDet = aug.to_deterministic()
        imgAug = augDet.augment_images([img] * 16)
        heatMapsAug = augDet.augment_heatmaps([heatMaps] * 16)
        
        imgAug = [heatMapsAug_i.draw_on_image(imgAug_i)[0]
                  for imgAug_i, heatMapsAug_i in zip(imgAug, heatMapsAug)]
        
        ia.imshow(ia.draw_grid(imgAug))


if __name__ == "__main__":
    main()