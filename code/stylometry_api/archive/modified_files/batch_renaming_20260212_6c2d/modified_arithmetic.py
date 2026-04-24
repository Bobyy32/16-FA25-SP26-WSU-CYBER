# Basic usage
aug = iaa.Add(value=50)
aug = iaa.GaussianBlur(sigma=1.0)
aug = iaa.Multiply(factor=1.5)

# With stochastic parameters
aug = iaa.Add(value=(-10, 10))  # Random value between -10 and 10
aug = iaa.GaussianBlur(sigma=(0.0, 2.0))  # Random sigma between 0 and 2

# Combining augmenters
seq = iaa.Sequential([
    iaa.GaussianBlur(sigma=1.0),
    iaa.Add(value=10),
    iaa.Multiply(factor=1.2)
])