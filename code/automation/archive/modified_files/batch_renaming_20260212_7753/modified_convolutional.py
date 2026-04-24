# Simple sharpening
sharpen = iaa.Sharpen(alpha=0.5)

# Edge detection with custom direction
directed_edges = iaa.DirectedEdgeDetect(alpha=0.7, direction=0.25)

# Gaussian blur
blur = iaa.GaussianBlur(sigma=1.0)

# Motion blur
motion_blur = iaa.MotionBlur(k=5, angle=45)