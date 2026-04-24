# Sharpen an image
aug = iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.7, 2.0))

# Detect edges from specific direction
aug = iaa.DirectedEdgeDetect(alpha=0.5, direction=0.25)  # 90 degrees

# Emboss effect
aug = iaa.Emboss(alpha=0.3, strength=(0.5, 1.0))