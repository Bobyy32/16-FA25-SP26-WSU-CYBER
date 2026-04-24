from imgaug.augmenters import MetaAugmenters
from imgaug.augmenters import MetaDtypeValidation

aug = MetaAugmenters(validate='int32')

# Example input array to be validated
array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# Try applying an augmenter (will raise if dtype not allowed)
try:
    output = aug.run(data=array)
except ValueError as e:
    print(f"Validation error: {e}")