Based on the provided imgaug Python code snippet, this module exports **image augmentation augmenters**, specifically:

## Main Exported Classes:

1. **`ElasticTransformation`** - Applies elastic transformations (simulating mesh-like distortions) to images, supporting various interpolation orders and backends (scipy/cv2).

2. **`Rot90`** - Efficiently rotates images by 90-degree multiples with optional size preservation.

3. **`WithPolarWarping`** - Applies augmenters in polar-transformed space, creating circular transformation effects.

4. **`Jigsaw`** - Implements jigsaw-pattern cell shuffling with configurable grid dimensions and movement steps.

## Module Purpose:
This is the `imgaug.augmenters` module that provides a comprehensive API for image augmentation tasks including geometric transformations, elastic warping, coordinate-based augmentations (keypoints, bounding boxes, polygons), and polar-coordinate space augmentations. The exports would typically be defined in an `__all__` list containing all public Augmenter classes for programmatic imports.