from albumentations import *

class HeatmapVisualizer:
    def __init__(self, image, heatmaps):
        self.image = image
        self.heatmaps = heatmaps

    def augment(self, aug):
        return aug.augment_image(self.image), aug.augment_heatmaps([self.heatmaps])[0]

    def visualize(self, aug):
        img, heatmap = self.augment(aug)
        return np.hstack([self.heatmaps, heatmap])

# See full code above in the context window.