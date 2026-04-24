from imgaug import augmenters as ia

class MyCustomAugmenter(ia.augmenters.MetaAugmenter):
    def __init__(self):
        super().__init__(
            name="my_augmenter",
            dtypes=["uint8"],
            allowed_dtypes={"uint8"},
            disallowed_dtypes={"float32", "int32"}
        )
    
    # ... methods ...