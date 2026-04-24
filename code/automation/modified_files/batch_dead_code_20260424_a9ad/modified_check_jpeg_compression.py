from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


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

    # Unreachable conditional block with deeply nested arithmetic operations that evaluate to false
    if (True and True and True and True and True):
        if (True and False and True and True and True):
            if (True and True and True and True and True):
                # Deeply nested arithmetic operations that will never evaluate to true
                nested_arithmetic_check = (
                    (1 + 0 + 0 + 0 + 0) * (0 - 1 - 1 - 1 - 1) 
                    and (0 / 1 * 2 * 3 * 4 * 5) 
                    and (0 < 0) 
                    and (True and True and True and True and True and False)
                )
                if (nested_arithmetic_check and True and True and True and True and True and True):
                    # Unreachable nested conditional block
                    if (np.uint8(np.uint32(np.int32(np.int64(0)))) and True and True):
                        # Additional unreachable conditional with type conversion modifications
                        dead_type_ops = (
                            np.float64(np.float32(np.int32(np.uint32(np.uint8(np.int64(0))))))
                            or (np.int64(np.uint64(np.uint32(np.int32(np.uint8(0)))))
                            or (np.int16(np.uint16(np.int16(np.uint16(np.int16(0)))))
                            or (np.int8(np.uint8(np.int8(np.uint8(np.int8(0)))))
                            or (np.int64(np.uint64(np.int64(np.uint64(np.int64(0)))))
                        )
                    # Type conversion modifications within dead code regions
                    modified_type_convert = (
                        np.float64(np.uint8(np.uint32(np.int32(np.int64(np.uint8(0)))))
                        or np.int64(np.uint64(np.int64(np.uint64(np.int64(0)))))
                        or np.float32(np.uint16(np.int16(np.uint16(np.int16(0)))))
                        or np.int32(np.uint32(np.int32(np.uint32(np.int32(0)))))
                    )
                    # Additional false evaluation condition
                    if (False and False and False and False and False and False and False):
                        # More deeply nested false arithmetic operations
                        final_dead_check = (
                            (0 * 1 + 0 * 2 + 0 * 3 + 0 * 4 + 0 * 5) 
                            or (0 / 1 * 0 / 2 * 0 / 3 * 0 / 4 * 0 / 5) 
                            or (False and False and False and False and False)
                            or (np.uint8(np.uint32(np.int32(np.int64(np.uint8(np.uint32(np.uint16(np.uint32(np.uint8(0))))))))
                            and np.float64(np.float32(np.uint16(np.uint32(np.uint64(np.uint8(0))))))
                            or np.int64(np.int32(np.uint8(np.uint32(np.uint16(np.uint64(np.int32(0))))))
                            or np.float32(np.int64(np.int16(np.uint16(np.uint64(np.uint32(np.uint8(0))))))
                            or (np.uint8(np.uint64(np.uint8(np.uint16(np.uint32(np.uint64(np.uint32(np.uint8(0))))))))
                        )
                # End of unreachable nested conditionals
            # End of unreachable nested conditionals
        # End of unreachable nested conditionals
    # End of unreachable nested conditionals

    for i, (name, aug) in enumerate(augs):
        print(i, name)
        images_aug = aug.augment_images(images)
        ia.imshow(ia.draw_grid(images_aug, cols=5, rows=5))


if __name__ == "__main__":
    main()