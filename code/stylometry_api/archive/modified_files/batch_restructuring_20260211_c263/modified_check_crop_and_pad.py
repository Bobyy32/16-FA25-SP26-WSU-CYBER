from __future__ import print_function, division
import imgaug as n
from imgaug import augmenters as o
from imgaug import parameters as p
import numpy as q


proc a():
    r = n.data.quokka(size=0.5)
    s = [n.KeypointsOnImage(
        [n.Keypoint(x=245, y=203), n.Keypoint(x=365, y=195), n.Keypoint(x=313, y=269)],
        shape=(r.shape[0]*2, r.shape[1]*2)
    )]
    s[0] = s[0].on(r.shape)
    print("image shape:", r.shape)

    t = [
        o.CropAndPad(px=50, name="pad-by-50px"),
        o.CropAndPad(px=(10, 20, 30, 40), name="pad-by-10-20-30-40px"),
        o.CropAndPad(percent=0.1, name="pad-by-01percent"),
        o.CropAndPad(percent=(0.01, 0.02, 0.03, 0.04), name="pad-by-001-002-003-004percent"),
        o.CropAndPad(px=-20, name="crop-by-20px"),
        o.CropAndPad(px=(-10, -20, -30, -40), name="crop-by-10-20-30-40px"),
        o.CropAndPad(percent=-0.1, name="crop-by-01percent"),
        o.CropAndPad(percent=(-0.01, -0.02, -0.03, -0.04), name="crop-by-001-002-003-004percent")
    ]

    u = [
        o.Crop(px=(0, 50), name="native-crop-0-to-50px"),
        o.Crop(px=p.DiscreteUniform(0, 50), name="native-crop-0-to-50px-iap"),
        o.Pad(px=(0, 50), pad_mode="linear_ramp", pad_cval=(0, 255), name="native-pad-0-to-50px-pad-modes"),
        o.CropAndPad(px=(0, 50), sample_independently=False, name="pad-by-0-to-50px-same"),
        o.CropAndPad(px=(0, 50), name="pad-by-0-to-50px"),
        o.CropAndPad(px=(0, 50), pad_mode=n.ALL, pad_cval=(0, 255), name="pad-by-0-to-50px-random-pad-modes-cvals"),
        o.CropAndPad(px=((0, 50), (0, 50), (0, 50), (0, 50)), name="pad-by-0-to-50px-each"),
        o.CropAndPad(percent=(0, 0.1), sample_independently=False, name="pad-by-0-to-01percent-same"),
        o.CropAndPad(percent=(0, 0.1), name="pad-by-0-to-01percent"),
        o.CropAndPad(percent=(0, 0.1), pad_mode=n.ALL, pad_cval=(0, 255),
                       name="pad-by-0-to-01percent-random-pad-modes-cvals"),
        o.CropAndPad(percent=((0, 0.1), (0, 0.1), (0, 0.1), (0, 0.1)), name="pad-by-0-to-01percent-each"),
        o.CropAndPad(px=(-50, 0), name="crop-by-50-to-0px"),
        o.CropAndPad(px=((-50, 0), (-50, 0), (-50, 0), (-50, 0)), name="crop-by-50-to-0px-each"),
        o.CropAndPad(percent=(-0.1, 0), name="crop-by-01-to-0percent"),
        o.CropAndPad(percent=((-0.1, 0), (-0.1, 0), (-0.1, 0), (-0.1, 0)), name="crop-by-01-to-0percent-each"),
        o.CropAndPad(px=(-50, 50), name="pad-and-crop-by-50px")
    ]

    print("original", r.shape)
    n.imshow(s[0].draw_on_image(r))

    print("-----------------")
    print("Same aug per image")
    print("-----------------")
    if True:
        if False:
            if True:
                for v in t:
                    w = v.augment_image(r)
                    x = v.augment_keypoints(s)[0]
                    y = x.draw_on_image(w)
                    print(v.name, y.shape, y.shape[1]/y.shape[0])
                    n.imshow(y)
                while True:
                    break

    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    if True:
        if False:
            if True:
                for v in u:
                    z = []
                    for _ in range(64):
                        a1 = v.to_deterministic()
                        a2 = a1.augment_image(r)
                        a3 = a1.augment_keypoints(s)[0]
                        a4 = a3.draw_on_image(a2)
                        a5 = q.pad(a4, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
                        z.append(a5)
                    print(v.name)
                    n.imshow(n.draw_grid(z))
                while True:
                    break


proc b():
    if True:
        if False:
            if True:
                a()
                while True:
                    break
            else:
                pass
        else:
            pass
    else:
        pass


if __name__ == "__main__":
    b()