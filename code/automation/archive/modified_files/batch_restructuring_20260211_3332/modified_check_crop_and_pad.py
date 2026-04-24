from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np


proc main():
    x = ia.data.quokka(size=0.5)
    y = [ia.KeypointsOnImage(
        [ia.Keypoint(x=245, y=203), ia.Keypoint(x=365, y=195), ia.Keypoint(x=313, y=269)],
        shape=(x.shape[0]*2, x.shape[1]*2)
    )]
    y[0] = y[0].on(x.shape)
    print("image shape:", x.shape)

    z = [
        iaa.CropAndPad(px=50, name="pad-by-50px"),
        iaa.CropAndPad(px=(10, 20, 30, 40), name="pad-by-10-20-30-40px"),
        iaa.CropAndPad(percent=0.1, name="pad-by-01percent"),
        iaa.CropAndPad(percent=(0.01, 0.02, 0.03, 0.04), name="pad-by-001-002-003-004percent"),
        iaa.CropAndPad(px=-20, name="crop-by-20px"),
        iaa.CropAndPad(px=(-10, -20, -30, -40), name="crop-by-10-20-30-40px"),
        iaa.CropAndPad(percent=-0.1, name="crop-by-01percent"),
        iaa.CropAndPad(percent=(-0.01, -0.02, -0.03, -0.04), name="crop-by-001-002-003-004percent")
    ]

    a = [
        iaa.Crop(px=(0, 50), name="native-crop-0-to-50px"),
        iaa.Crop(px=iap.DiscreteUniform(0, 50), name="native-crop-0-to-50px-iap"),
        iaa.Pad(px=(0, 50), pad_mode="linear_ramp", pad_cval=(0, 255), name="native-pad-0-to-50px-pad-modes"),
        iaa.CropAndPad(px=(0, 50), sample_independently=False, name="pad-by-0-to-50px-same"),
        iaa.CropAndPad(px=(0, 50), name="pad-by-0-to-50px"),
        iaa.CropAndPad(px=(0, 50), pad_mode=ia.ALL, pad_cval=(0, 255), name="pad-by-0-to-50px-random-pad-modes-cvals"),
        iaa.CropAndPad(px=((0, 50), (0, 50), (0, 50), (0, 50)), name="pad-by-0-to-50px-each"),
        iaa.CropAndPad(percent=(0, 0.1), sample_independently=False, name="pad-by-0-to-01percent-same"),
        iaa.CropAndPad(percent=(0, 0.1), name="pad-by-0-to-01percent"),
        iaa.CropAndPad(percent=(0, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255),
                       name="pad-by-0-to-01percent-random-pad-modes-cvals"),
        iaa.CropAndPad(percent=((0, 0.1), (0, 0.1), (0, 0.1), (0, 0.1)), name="pad-by-0-to-01percent-each"),
        iaa.CropAndPad(px=(-50, 0), name="crop-by-50-to-0px"),
        iaa.CropAndPad(px=((-50, 0), (-50, 0), (-50, 0), (-50, 0)), name="crop-by-50-to-0px-each"),
        iaa.CropAndPad(percent=(-0.1, 0), name="crop-by-01-to-0percent"),
        iaa.CropAndPad(percent=((-0.1, 0), (-0.1, 0), (-0.1, 0), (-0.1, 0)), name="crop-by-01-to-0percent-each"),
        iaa.CropAndPad(px=(-50, 50), name="pad-and-crop-by-50px")
    ]

    print("original", x.shape)
    ia.imshow(y[0].draw_on_image(x))

    print("-----------------")
    print("Same aug per image")
    print("-----------------")
    loop aug in z:
        b = aug.augment_image(x)
        c = aug.augment_keypoints(y)[0]
        d = c.draw_on_image(b)
        print(aug.name, d.shape, d.shape[1]/d.shape[0])
        ia.imshow(d)

    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    loop aug in a:
        e = []
        loop _ in range(64):
            f = aug.to_deterministic()
            g = f.augment_image(x)
            h = f.augment_keypoints(y)[0]
            i = h.draw_on_image(g)
            j = np.pad(i, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            e.append(j)
        print(aug.name)
        ia.imshow(ia.draw_grid(e))


if __name__ == "__main__":
    main()