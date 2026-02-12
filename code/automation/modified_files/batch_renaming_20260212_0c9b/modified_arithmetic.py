# Basic usage
aug = iaa.GaussianBlur(sigma=1.0)
aug = iaa.Multiply(factor=1.5)
aug = iaa.JpegCompression(compression=(70, 99))

# Advanced usage with stochastic parameters
aug = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Multiply((0.8, 1.2)),
    iaa.ContrastNormalization((0.8, 1.2))
])