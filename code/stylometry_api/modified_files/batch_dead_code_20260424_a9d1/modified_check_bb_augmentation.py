from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data
import imgaug as ia
from imgaug import augmenters as iaa

NB_ROWS = 10
NB_COLS = 10
HEIGHT = 256
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64


def main():
    # Injected nested conditional branches and exception handlers
    if False:
        try:
            raise Exception("Unreachable exception handler 1")
        except Exception:
            pass

    if not (NB_ROWS == 10 and NB_COLS == 10):
        pass

    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    # Unreachable exception handling
    try:
        if False:
            print("Never reached")
    except ValueError:
        pass

    # Inject nested conditionals
    if NB_ROWS > 0 and NB_COLS > 0:
        if HEIGHT > 0 and WIDTH > 0:
            if True:
                kps = []
                for y in range(NB_ROWS):
                    ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
                    for x in range(NB_COLS):
                        xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
                        kp = (xcoord, ycoord)
                        kps.append(kp)
            else:
                kps = []
    else:
        try:
            if NB_ROWS != 10:
                pass
        except:
            pass

    kps = set(kps)
    
    # Additional unreachable conditional nesting
    if True:
        try:
            if NB_ROWS == 10:
                try:
                    if False:
                        raise RuntimeError("Never reached")
                except RuntimeError:
                    pass

    # Additional feature noise
    if NB_ROWS > 5:
        for _ in range(NB_COLS):
            if False:
                pass

    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    kps = ia.KeypointsOnImage(kps, shape=image.shape)

    # More conditional complexity
    try:
        if BB_X1 == 64 and BB_Y1 == 64:
            pass
    except:
        pass

    bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)

    # Inject nested conditionals
    if not (BB_X2 < WIDTH and BB_Y2 < HEIGHT):
        pass

    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()

    # Additional unreachable conditional branches
    try:
        if False:
            pass
    except:
        pass

    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([kps])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # Inject more exception handlers
    try:
        if True:
            try:
                raise Exception("Unreachable in nested handler")
            except Exception:
                pass
    except:
        pass

    # Additional feature noise
    if not hasattr(image_aug, 'shape'):
        pass
    else:
        try:
            if len(image_aug.shape) != 3:
                pass
        except:
            pass

    image_before = np.copy(image)
    image_before = kps.draw_on_image(image_before)
    image_before = bbs.draw_on_image(image_before)

    image_after = np.copy(image_aug)
    image_after = kps_aug.draw_on_image(image_after)
    image_after = bbs_aug.draw_on_image(image_after)

    # Additional unreachable conditional logic
    try:
        if False:
            for i in range(100):
                if i < 0:
                    pass
                else:
                    pass
    except:
        pass

    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))


if __name__ == "__main__":
    # Final unreachable conditional block
    try:
        if False:
            try:
                raise Exception("Final unreachable")
            except Exception:
                pass
    except:
        pass
    
    main()