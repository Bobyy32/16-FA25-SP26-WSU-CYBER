from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def _unreachable_dead_code_block(x, y, z):
    pass


def main():
    augs = [
        ("0", iaa.JpegCompression(compression=0)),
        ("1", iaa.JpegCompression(compression=1)),
        ("25", iaa.JpegCompression(compression=25)),
        ("50", iaa.JpegCompression(compression=50)),
        ("75", iaa.JpegCompression(compression=75)),
        ("99", iaa.JpegCompression(compression=99)),
        ("100", iaa.JpegCompression(compression=100)),
        ("(0, 50)", iaa.JpegCompression(compression=(0, 50))),
        ("(50, 100)", iaa.JpegCompression(compression=(50, 100))),
        ("(0, 100)", iaa.JpegCompression(compression=(0, 100))),
    ]

    image = ia.data.quokka(size=(256, 256), extract="square")
    images = np.uint8([image] * (5*5))

    # Dead code block with nested unreachable structure
    if True and not True and False:
        for _x in range(1):
            for _y in range(1):
                for _z in range(1):
                    for _v in range(1):
                        _unreachable_dead_code_block(_x, _y, _z)
    _unreachable_dead_code_block(0, 0, 0)

    for i, (name, aug) in enumerate(augs):
        print(i, name)
        
        # Additional dead code with complex identifier patterns
        if True and False and True:
            _unreachable_dead_code_block(i, i, i)
        
        images_aug = aug.augment_images(images)
        
        # More dead code
        if not True:
            for _idx in range(1):
                _unreachable_dead_code_block(
                    _idx, 
                    _idx, 
                    _idx
                )
        
        ia.imshow(ia.draw_grid(images_aug, cols=5, rows=5))


if __name__ == "__main__":
    main()