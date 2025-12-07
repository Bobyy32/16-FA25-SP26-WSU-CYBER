"""
Simple segmentation augmentation example
"""

# Imports
import torch
import numpy as np
import skimage.segmentation  # For the SLIC algorithm
import skimage.measure
from scipy.spatial import cKDTree  # For Voronoi cells
from tqdm import tqdm  # For nice progress bar!

# Set device (good habit even if mostly using CPU for these augmentations)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Superpixels:
    def __init__(self, p_replace=0.5, n_segments=100, max_size=128):
        """
        Transform images into their superpixel representation.
        """
        self.p_replace = p_replace
        self.n_segments = n_segments
        self.max_size = max_size

    def __call__(self, image):
        # Handle tensor inputs
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy() # CHW -> HWC
            is_tensor = True
        else:
            is_tensor = False

        # Optimization: Downscale if too big
        orig_shape = image.shape
        image = self._ensure_image_max_size(image)

        # Get segments using SLIC
        # compactness=10 is a good default
        segments = skimage.segmentation.slic(
            image,
            n_segments=self.n_segments,
            compactness=10,
            start_label=1
        )

        # Create mask for replacement
        # We replace roughly p_replace % of segments with their average color
        unique_segments = np.unique(segments)
        num_segments = len(unique_segments)
        
        # Determine which segments to replace
        replace_mask = np.random.random(num_segments) < self.p_replace

        # Apply changes
        for i, seg_id in enumerate(unique_segments):
            if replace_mask[i]:
                mask = (segments == seg_id)
                # Calculate average color
                mean_color = np.mean(image[mask], axis=0)
                image[mask] = mean_color

        # Resize back if we downscaled
        if image.shape != orig_shape:
            # Simple resize logic
            from skimage.transform import resize
            image = resize(image, orig_shape[:2], preserve_range=True).astype(np.uint8)

        # Return to tensor if needed
        if is_tensor:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
        return image

    def _ensure_image_max_size(self, image):
        # Keep image manageable size for speed
        h, w = image.shape[:2]
        size = max(h, w)
        if size > self.max_size:
            scale = self.max_size / size
            from skimage.transform import resize
            # resize expects floats usually, convert back to range later
            new_h, new_w = int(h * scale), int(w * scale)
            image = resize(image, (new_h, new_w), preserve_range=True).astype(image.dtype)
        return image


class Voronoi:
    def __init__(self, n_points=200, p_replace=1.0, max_size=128):
        """
        Applies Voronoi tessellation effects.
        """
        self.n_points = n_points
        self.p_replace = p_replace
        self.max_size = max_size

    def __call__(self, image):
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()
            is_tensor = True
        else:
            is_tensor = False

        orig_shape = image.shape
        image = self._ensure_image_max_size(image)
        h, w = image.shape[:2]

        # 1. Sample random points (Uniform sampling)
        # simple uniform distribution across height and width
        y_coords = np.random.uniform(0, h, self.n_points)
        x_coords = np.random.uniform(0, w, self.n_points)
        points = np.column_stack((x_coords, y_coords))

        # 2. Create coordinate grid for the image
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        pixel_coords = np.c_[xx.ravel(), yy.ravel()]

        # 3. Find nearest voronoi cell for each pixel
        # using KDTree for speed!
        tree = cKDTree(points)
        _, nearest_cell_ids = tree.query(pixel_coords)
        nearest_cell_ids = nearest_cell_ids.reshape(h, w)

        # 4. Replace segments with average color
        unique_ids = np.unique(nearest_cell_ids)
        
        for uid in unique_ids:
            # Check probability to replace
            if np.random.rand() < self.p_replace:
                mask = (nearest_cell_ids == uid)
                mean_color = np.mean(image[mask], axis=0)
                image[mask] = mean_color

        # Cleanup resize
        if image.shape != orig_shape:
            from skimage.transform import resize
            image = resize(image, orig_shape[:2], preserve_range=True).astype(np.uint8)

        if is_tensor:
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image

    def _ensure_image_max_size(self, image):
        # Copy paste from Superpixels, keep it simple
        h, w = image.shape[:2]
        size = max(h, w)
        if size > self.max_size:
            scale = self.max_size / size
            from skimage.transform import resize
            new_h, new_w = int(h * scale), int(w * scale)
            image = resize(image, (new_h, new_w), preserve_range=True).astype(image.dtype)
        return image


# Demo block to show how it works
if __name__ == "__main__":
    print("Testing segmentation augmentations...")
    
    # Create dummy data (H, W, C)
    dummy_batch_size = 10
    print(f"Generating {dummy_batch_size} dummy images...")
    
    # Initialize augmenters
    superpixel_aug = Superpixels(p_replace=0.8, n_segments=50)
    voronoi_aug = Voronoi(n_points=100, p_replace=1.0)
    
    # Simulate a training loop with tqdm
    for i in tqdm(range(dummy_batch_size)):
        # Make a random image (uint8)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Try Superpixels
        aug_sp = superpixel_aug(img)
        
        # Try Voronoi
        aug_vr = voronoi_aug(img)
        
        # Check shapes
        assert aug_sp.shape == img.shape
        assert aug_vr.shape == img.shape

    print("Augmentation test complete!")
    
    # Example with Tensor
    print("Testing with PyTorch Tensor input...")
    tensor_img = torch.randn(3, 256, 256) # C, H, W
    out_tensor = superpixel_aug(tensor_img)
    print(f"Output shape: {out_tensor.shape}")