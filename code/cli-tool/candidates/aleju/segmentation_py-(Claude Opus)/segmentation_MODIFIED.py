"""
Simple segmentation augmenters for image processing
"""

# Imports
from __future__ import print_function, division, absolute_import
from abc import ABCMeta, abstractmethod
import numpy as np
import skimage.segmentation  # Use full path to avoid confusion
import skimage.measure
import six
import six.moves as sm
import imgaug as ia
from . import meta  # Local imports
from .. import random as iarandom
from .. import parameters as iap
from .. import dtypes as iadt
from ..imgaug import _NUMBA_INSTALLED, _numbajit  # For performance optimization


# Check skimage version for SLIC support
_SLIC_SUPPORTS_START_LABEL = (
    tuple(map(int, skimage.__version__.split(".")[0:2])) >= (0, 17)
)  # Added in 0.5.0


# Helper function to ensure images don't exceed max size
def _ensure_image_max_size(image, max_size, interpolation):
    # Downscale if any side violates max_size
    # Keep aspect ratio when downscaling
    if max_size is not None:
        size = max(image.shape[0], image.shape[1])
        if size > max_size:
            resize_factor = max_size / size
            new_height = int(image.shape[0] * resize_factor)
            new_width = int(image.shape[1] * resize_factor)
            image = ia.imresize_single_image(
                image,
                (new_height, new_width),
                interpolation=interpolation)
    return image


# Simple Superpixels class for SLIC algorithm
class Superpixels(meta.Augmenter):
    # Transform images to superpixel representation using SLIC
    # Note: This is fairly slow
    
    def __init__(self, p_replace=0.5, n_segments=100, max_size=128,
                 interpolation="cubic", seed=None, name=None,
                 random_state=None, deterministic=False):
        super(Superpixels, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        
        # Handle probability parameter
        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True,
            list_to_choice=True)
        
        # Handle segments parameter
        self.n_segments = iap.handle_discrete_param(
            n_segments, "n_segments", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        
        self.max_size = max_size
        self.interpolation = interpolation

    def get_parameters(self):
        return [self.p_replace, self.n_segments]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        # Standard batch augmentation
        if batch.images is not None:
            batch.images = self._augment_images(
                batch.images, random_state, parents, hooks)
        
        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps(
                batch.heatmaps, random_state, parents, hooks, "arr_0to1")
        
        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps(
                batch.segmentation_maps, random_state, parents, hooks, "arr")
        
        return batch

    def _augment_images(self, images, random_state, parents, hooks):
        # Main augmentation logic
        output = images
        nb_images = len(images)
        rss = random_state.duplicate(nb_images)
        
        for i, (image, rs) in enumerate(zip(images, rss)):
            output[i] = self._augment_single_image(image, rs, hooks)
        
        return output

    def _augment_single_image(self, image, random_state, hooks):
        # Process single image
        orig_shape = image.shape
        orig_dtype = image.dtype
        
        # Downscale if needed
        image_small = _ensure_image_max_size(
            image, self.max_size, self.interpolation)
        
        # Get parameters for this image
        n_segments = self.n_segments.draw_sample(random_state)
        n_segments = max(n_segments, 1)
        
        # Apply SLIC algorithm
        image_sp = self._apply_slic(image_small, n_segments, random_state)
        
        # Resize back to original if needed
        if orig_shape[0:2] != image_sp.shape[0:2]:
            image_sp = ia.imresize_single_image(
                image_sp, orig_shape[0:2], interpolation=self.interpolation)
        
        # Check if we need to replace segments
        if self._is_replace_samples:
            p_replace_samples = self.p_replace.draw_samples(
                (n_segments,), random_state)
            replace_samples = (p_replace_samples > 0.5)
        else:
            p_replace_sample = self.p_replace.draw_sample(random_state)
            replace_samples = (p_replace_sample > 0.5)
        
        # Replace segments with average color if needed
        if np.any(replace_samples):
            image_sp = self._replace_segments(
                image_small, image_sp, segments, replace_samples, orig_shape)
        
        return image_sp.astype(orig_dtype)

    def _apply_slic(self, image, n_segments, random_state):
        # Simple SLIC implementation
        seed = random_state.generate_seed_()
        
        # SLIC parameters
        segments = skimage.segmentation.slic(
            image,
            n_segments=n_segments,
            compactness=10,
            max_iter=10,
            sigma=0,
            max_size_factor=3,
            slic_zero=False,
            start_label=0 if _SLIC_SUPPORTS_START_LABEL else None,
            mask=None,
            convert2lab=True,
            enforce_connectivity=False,
            min_size_factor=None)
        
        return segments

    def _replace_segments(self, image, image_sp, segments, replace_mask, 
                         orig_shape):
        # Replace segments with their average colors
        if not hasattr(skimage.measure, "regionprops"):
            # Fallback for old skimage versions
            return self._replace_segments_slow(
                image, segments, replace_mask, orig_shape)
        
        # Fast version using regionprops
        regions = skimage.measure.regionprops(segments+1, intensity_image=image)
        
        for region in regions:
            if replace_mask[region.label-1]:
                mean_color = region.mean_intensity
                image_sp[segments == region.label-1] = mean_color
        
        return image_sp


# Voronoi base class
@six.add_metaclass(ABCMeta)
class _AbstractVoronoi(meta.Augmenter):
    # Base voronoi implementation
    
    def __init__(self, points_sampler, p_drop_points=0.4,
                 p_replace=1.0, max_size=128,
                 interpolation="linear", seed=None, name=None,
                 random_state=None, deterministic=False):
        super(_AbstractVoronoi, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        
        # Check points sampler
        assert isinstance(points_sampler, IPointsSampler), (
            "Expected IPointsSampler, got %s" % type(points_sampler))
        
        # Set up dropout if needed
        if p_drop_points is not None and p_drop_points > 0:
            self.points_sampler = DropoutPointsSampler(
                points_sampler, p_drop_points)
        else:
            self.points_sampler = points_sampler
        
        # Handle replace probability
        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True,
            list_to_choice=True)
        
        self.max_size = max_size
        self.interpolation = interpolation

    def get_parameters(self):
        return [self.points_sampler, self.p_replace]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        # Standard batch processing
        if batch.images is not None:
            batch.images = self._augment_images(
                batch.images, random_state, parents, hooks)
        return batch

    def _augment_images(self, images, random_state, parents, hooks):
        # Process all images
        output = images
        rss = random_state.duplicate(len(images) + 1)
        
        # Sample points for all images
        points_on_images = self.points_sampler.sample_points(images, rss[-1])
        
        # Apply voronoi to each
        for i, (image, points, rs) in enumerate(
                zip(images, points_on_images, rss[:-1])):
            output[i] = self._augment_single_image(image, points, rs, hooks)
        
        return output

    def _augment_single_image(self, image, points, random_state, hooks):
        # Apply voronoi to single image
        orig_shape = image.shape
        orig_dtype = image.dtype
        
        # Downscale if needed
        image_small = _ensure_image_max_size(
            image, self.max_size, self.interpolation)
        
        # Scale points if image was resized
        if image_small.shape[0:2] != orig_shape[0:2]:
            points = self._scale_points(points, orig_shape, image_small.shape)
        
        # Create voronoi
        image_voronoi = self._create_voronoi(image_small, points)
        
        # Decide replacement
        p_replace = self.p_replace.draw_sample(random_state)
        if p_replace > 0.5:
            image_result = image_voronoi
        else:
            # Mix original and voronoi based on probability
            mask = random_state.uniform(0, 1, image_small.shape[0:2]) < p_replace
            mask = np.stack([mask] * image_small.shape[2], axis=-1)
            image_result = np.where(mask, image_voronoi, image_small)
        
        # Resize back if needed
        if image_result.shape[0:2] != orig_shape[0:2]:
            image_result = ia.imresize_single_image(
                image_result, orig_shape[0:2], 
                interpolation=self.interpolation)
        
        return image_result.astype(orig_dtype)

    @abstractmethod
    def _create_voronoi(self, image, points):
        # Abstract method - implement in subclasses
        pass

    def _scale_points(self, points, orig_shape, new_shape):
        # Scale points to match resized image
        height_scale = new_shape[0] / orig_shape[0]
        width_scale = new_shape[1] / orig_shape[1]
        
        points_scaled = points.copy()
        points_scaled[:, 0] *= width_scale
        points_scaled[:, 1] *= height_scale
        
        return points_scaled


# Regular Voronoi implementation
class Voronoi(_AbstractVoronoi):
    # Standard voronoi with custom point sampling
    
    def __init__(self, points_sampler, p_drop_points=0.4,
                 p_replace=(0.0, 1.0), max_size=128,
                 interpolation="linear", seed=None, name=None,
                 random_state=None, deterministic=False):
        super(Voronoi, self).__init__(
            points_sampler=points_sampler,
            p_drop_points=p_drop_points,
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic)
    
    def _create_voronoi(self, image, points):
        # Create voronoi cells
        if len(points) == 0:
            return image
        
        # Simple voronoi using distance transform
        height, width = image.shape[0:2]
        cells = np.zeros((height, width), dtype=np.int32)
        
        # For each pixel, find closest point
        yy, xx = np.mgrid[0:height, 0:width]
        
        for i, point in enumerate(points):
            px, py = int(point[0]), int(point[1])
            dist = (xx - px) ** 2 + (yy - py) ** 2
            
            if i == 0:
                min_dist = dist
                cells = np.zeros_like(dist, dtype=np.int32)
            else:
                mask = dist < min_dist
                cells[mask] = i
                min_dist = np.minimum(min_dist, dist)
        
        # Color cells with average
        result = np.zeros_like(image)
        for i in range(len(points)):
            mask = cells == i
            if np.any(mask):
                # Get average color for this cell
                avg_color = np.mean(image[mask], axis=0)
                result[mask] = avg_color
        
        return result


# Uniform Voronoi - uses uniform point sampling
class UniformVoronoi(_AbstractVoronoi):
    # Voronoi with uniformly sampled points
    
    def __init__(self, n_points=(20, 50), p_drop_points=0.0,
                 p_replace=1.0, max_size=128,
                 interpolation="linear", seed=None, name=None,
                 random_state=None, deterministic=False):
        # Create uniform sampler
        points_sampler = UniformPointsSampler(n_points)
        
        super(UniformVoronoi, self).__init__(
            points_sampler=points_sampler,
            p_drop_points=p_drop_points,
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic)
        
        self.n_points = self.points_sampler.other_points_sampler.n_points \
            if isinstance(self.points_sampler, DropoutPointsSampler) \
            else self.points_sampler.n_points
    
    def _create_voronoi(self, image, points):
        # Same as parent
        return Voronoi._create_voronoi(self, image, points)

    def get_parameters(self):
        return [self.n_points, self.p_replace]


# Grid-based Voronoi
class RegularGridVoronoi(_AbstractVoronoi):
    # Voronoi with regular grid sampling
    
    def __init__(self, n_rows=(10, 30), n_cols=(10, 30),
                 p_drop_points=0.0, p_replace=1.0,
                 max_size=128, interpolation="linear",
                 seed=None, name=None, random_state=None,
                 deterministic=False):
        # Create grid sampler
        points_sampler = RegularGridPointsSampler(n_rows, n_cols)
        
        super(RegularGridVoronoi, self).__init__(
            points_sampler=points_sampler,
            p_drop_points=p_drop_points,
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic)
        
        # Store grid params for easy access
        if isinstance(self.points_sampler, DropoutPointsSampler):
            self.n_rows = self.points_sampler.other_points_sampler.n_rows
            self.n_cols = self.points_sampler.other_points_sampler.n_cols
        else:
            self.n_rows = self.points_sampler.n_rows
            self.n_cols = self.points_sampler.n_cols
    
    def _create_voronoi(self, image, points):
        # Same as parent
        return Voronoi._create_voronoi(self, image, points)

    def get_parameters(self):
        return [self.n_rows, self.n_cols, self.p_replace]


# Relative grid Voronoi
class RelativeRegularGridVoronoi(_AbstractVoronoi):
    # Grid with relative spacing
    
    def __init__(self, n_rows_frac=(0.05, 0.15), n_cols_frac=(0.05, 0.15),
                 p_drop_points=0.0, p_replace=1.0,
                 max_size=128, interpolation="linear",
                 seed=None, name=None, random_state=None,
                 deterministic=False):
        # Create relative grid sampler
        points_sampler = RelativeRegularGridPointsSampler(
            n_rows_frac, n_cols_frac)
        
        super(RelativeRegularGridVoronoi, self).__init__(
            points_sampler=points_sampler,
            p_drop_points=p_drop_points,
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic)
        
        # Store params
        if isinstance(self.points_sampler, DropoutPointsSampler):
            self.n_rows_frac = self.points_sampler.other_points_sampler.n_rows_frac
            self.n_cols_frac = self.points_sampler.other_points_sampler.n_cols_frac
        else:
            self.n_rows_frac = self.points_sampler.n_rows_frac
            self.n_cols_frac = self.points_sampler.n_cols_frac
    
    def _create_voronoi(self, image, points):
        # Same as parent
        return Voronoi._create_voronoi(self, image, points)

    def get_parameters(self):
        return [self.n_rows_frac, self.n_cols_frac, self.p_replace]


# Point sampler interface
@six.add_metaclass(ABCMeta)
class IPointsSampler(object):
    # Interface for point sampling
    
    @abstractmethod
    def sample_points(self, images, random_state):
        # Must return list of points for each image
        pass


# Helper to verify images
def _verify_sample_points_images(images):
    # Simple validation
    assert isinstance(images, list), "Expected list of images"
    assert len(images) > 0, "Need at least one image"


# Grid points sampler
class RegularGridPointsSampler(IPointsSampler):
    # Sample points on regular grid
    
    def __init__(self, n_rows, n_cols=None):
        self.n_rows = iap.handle_discrete_param(
            n_rows, "n_rows", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True,
            allow_floats=False)
        
        if n_cols is None:
            self.n_cols = self.n_rows
        else:
            self.n_cols = iap.handle_discrete_param(
                n_cols, "n_cols", value_range=(1, None),
                tuple_to_uniform=True, list_to_choice=True,
                allow_floats=False)
    
    def sample_points(self, images, random_state):
        # Generate grid points
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)
        
        rss = random_state.duplicate(2 * len(images))
        points_list = []
        
        for i, image in enumerate(images):
            height, width = image.shape[0:2]
            
            # Sample grid size
            n_rows = self.n_rows.draw_sample(rss[2*i])
            n_cols = self.n_cols.draw_sample(rss[2*i + 1])
            
            # Create grid
            points = self._create_grid_points(height, width, n_rows, n_cols)
            points_list.append(points)
        
        return points_list
    
    def _create_grid_points(self, height, width, n_rows, n_cols):
        # Create evenly spaced grid
        n_rows = max(n_rows, 1)
        n_cols = max(n_cols, 1)
        
        y_coords = np.linspace(0, height, n_rows)
        x_coords = np.linspace(0, width, n_cols)
        
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        
        return points


# Relative grid sampler
class RelativeRegularGridPointsSampler(IPointsSampler):
    # Grid with relative spacing
    
    def __init__(self, n_rows_frac, n_cols_frac=None):
        self.n_rows_frac = iap.handle_continuous_param(
            n_rows_frac, "n_rows_frac", value_range=(0.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        
        if n_cols_frac is None:
            self.n_cols_frac = self.n_rows_frac
        else:
            self.n_cols_frac = iap.handle_continuous_param(
                n_cols_frac, "n_cols_frac", value_range=(0.0, 1.0),
                tuple_to_uniform=True, list_to_choice=True)
    
    def sample_points(self, images, random_state):
        # Generate relative grid
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)
        
        rss = random_state.duplicate(2 * len(images))
        points_list = []
        
        for i, image in enumerate(images):
            height, width = image.shape[0:2]
            
            # Sample fractions
            row_frac = self.n_rows_frac.draw_sample(rss[2*i])
            col_frac = self.n_cols_frac.draw_sample(rss[2*i + 1])
            
            # Convert to absolute
            n_rows = max(int(row_frac * height), 1)
            n_cols = max(int(col_frac * width), 1)
            
            # Create grid
            grid_sampler = RegularGridPointsSampler(n_rows, n_cols)
            points = grid_sampler._create_grid_points(height, width, n_rows, n_cols)
            points_list.append(points)
        
        return points_list


# Dropout points sampler
class DropoutPointsSampler(IPointsSampler):
    # Randomly drop points from another sampler
    
    def __init__(self, other_points_sampler, p_drop):
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected IPointsSampler, got %s" % type(other_points_sampler))
        
        self.other_points_sampler = other_points_sampler
        self.p_drop = iap.handle_probability_param(
            p_drop, "p_drop", tuple_to_uniform=True,
            list_to_choice=True)
    
    def sample_points(self, images, random_state):
        # Sample then drop
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)
        
        rss = random_state.duplicate(len(images) + 1)
        
        # Get points from other sampler
        points_on_images = self.other_points_sampler.sample_points(
            images, rss[-1])
        
        # Apply dropout
        keep_masks = [self._draw_samples_for_image(points, rs)
                     for points, rs in zip(points_on_images, rss[:-1])]
        
        return self._apply_dropout_masks(points_on_images, keep_masks)
    
    def _draw_samples_for_image(self, points_on_image, random_state):
        # Decide which points to keep
        drop_samples = self.p_drop.draw_samples(
            (len(points_on_image),), random_state)
        keep_mask = (drop_samples <= 0.5)  # Keep if below threshold
        return keep_mask
    
    @classmethod
    def _apply_dropout_masks(cls, points_on_images, keep_masks):
        # Apply masks to points
        points_dropped = []
        
        for points, keep_mask in zip(points_on_images, keep_masks):
            if len(points) == 0:
                # No points to drop
                poi_dropped = points
            else:
                if not np.any(keep_mask):
                    # Keep at least one point
                    idx = (len(points) - 1) // 2
                    keep_mask = np.copy(keep_mask)
                    keep_mask[idx] = True
                poi_dropped = points[keep_mask, :]
            points_dropped.append(poi_dropped)
        
        return points_dropped


# Uniform points sampler
class UniformPointsSampler(IPointsSampler):
    # Sample points uniformly
    
    def __init__(self, n_points):
        self.n_points = iap.handle_discrete_param(
            n_points, "n_points", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True,
            allow_floats=False)
    
    def sample_points(self, images, random_state):
        # Sample uniform points
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)
        
        rss = random_state.duplicate(2)
        
        # Draw number of points per image
        n_points_imagewise = self._draw_samples(len(images), rss[0])
        
        # Generate relative coordinates
        n_points_total = np.sum(n_points_imagewise)
        n_components_total = 2 * n_points_total
        coords_relative = rss[1].uniform(0.0, 1.0, n_components_total)
        coords_relative_xy = coords_relative.reshape(n_points_total, 2)
        
        # Convert to absolute
        return self._convert_relative_coords_to_absolute(
            coords_relative_xy, n_points_imagewise, images)
    
    def _draw_samples(self, n_augmentables, random_state):
        # Draw n_points for each image
        n_points = self.n_points.draw_samples(
            (n_augmentables,), random_state=random_state)
        n_points_clipped = np.clip(n_points, 1, None)
        return n_points_clipped
    
    @classmethod
    def _convert_relative_coords_to_absolute(cls, coords_rel_xy,
                                            n_points_imagewise, images):
        # Convert relative to absolute coordinates
        coords_absolute = []
        i = 0
        
        for image, n_points_image in zip(images, n_points_imagewise):
            height, width = image.shape[0:2]
            xx = coords_rel_xy[i:i+n_points_image, 0]
            yy = coords_rel_xy[i:i+n_points_image, 1]
            
            # Scale and clip
            xx_int = np.clip(np.round(xx * width), 0, width)
            yy_int = np.clip(np.round(yy * height), 0, height)
            
            coords_absolute.append(np.stack([xx_int, yy_int], axis=-1))
            i += n_points_image
        
        return coords_absolute


# Subsampling points sampler
class SubsamplingPointsSampler(IPointsSampler):
    # Limit maximum number of points
    
    def __init__(self, other_points_sampler, n_points_max):
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected IPointsSampler, got %s" % type(other_points_sampler))
        
        self.other_points_sampler = other_points_sampler
        self.n_points_max = np.clip(n_points_max, -1, None)
        
        if self.n_points_max == 0:
            ia.warn("n_points_max=0 will result in no points")
    
    def sample_points(self, images, random_state):
        # Sample then subsample if needed
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)
        
        rss = random_state.duplicate(len(images) + 1)
        
        # Get points
        points_on_images = self.other_points_sampler.sample_points(
            images, rss[-1])
        
        # Subsample each
        return [self._subsample(points, self.n_points_max, rs)
                for points, rs in zip(points_on_images, rss[:-1])]
    
    @classmethod
    def _subsample(cls, points_on_image, n_points_max, random_state):
        # Randomly select subset if too many points
        if len(points_on_image) <= n_points_max:
            return points_on_image
        
        indices = np.arange(len(points_on_image))
        indices_to_keep = random_state.permutation(indices)[0:n_points_max]
        return points_on_image[indices_to_keep]


# TODO: Add more samplers
# - Poisson disc sampling for better distribution
# - Jittered grid for slight randomness
# - Clustered points for grouped effects