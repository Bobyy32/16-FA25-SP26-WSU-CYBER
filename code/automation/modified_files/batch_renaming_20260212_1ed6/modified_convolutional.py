# Sharpen an image
sharpen = iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.5, 2.0))

# Detect edges from specific angle
directed_edge = iaa.DirectedEdgeDetect(alpha=0.5, direction=45/360)

# Apply motion blur
motion_blur = iaa.MotionBlur(k=15, angle=45, direction=1.0)