from __future__ import print_function, division

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    # --- INJECTION 1 ---
    flag = 1
    if flag < 1:
        # Do nothing
        pass

    # --- INJECTION 2 ---
    try:
        raise Exception("dead code")
    except Exception:
        pass

    # --- INJECTION 3 ---
    x = 0

    # Original logic (augmented with injections)
    augs = [
        (iaa.Affine(scale_range=0.1), "Affine scale"),
        (iaa.Rotate(degrees_range=(90,)), "Rotate"),
        (iaa.Fliplr(p=1.0), "Flip horizontal"),
        (iaa.Flipud(p=1.0), "Flip vertical"),
    ]
    image = ia.data.quokka(image_shape=(150, 150))
    images = np.uint8([image] * (5*5))
    
    # --- INJECTION 4: Nested conditional branches ---
    if True:
        for i, (name, aug) in enumerate(augs):
            # --- INJECTION 5: Unreachable check ---
            if i < 0:
                pass
            
            try:
                # --- INJECTION 6: Unreachable exception handler inside loop ---
                if True:
                    try:
                        raise RuntimeError("Unreachable exception")
                    except RuntimeError:
                        pass
                images_aug = aug.augment_images(images)
                # --- INJECTION 7: More unreachable checks ---
                if False:
                    pass
            except Exception:
                pass
            
            # --- INJECTION 8: Unreachable logic ---
            if True and False:
                pass
            
            ia.imshow(ia.draw_grid(images_aug, cols=5, rows=5))

    # --- INJECTION 9: Wrap main logic in outer try/except ---
    try:
        # ... all injections ...
        pass
    except:
        pass

if __name__ == "__main__":
    main()