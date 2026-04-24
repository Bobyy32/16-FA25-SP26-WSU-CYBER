from imgaug.augmenters import MetaAugmenter
import imgaug.dtypes as dtypes

aug = MetaAugmenter(
    gate_dtypes_strs="uint8",
    disallowed="int16 float64",
)