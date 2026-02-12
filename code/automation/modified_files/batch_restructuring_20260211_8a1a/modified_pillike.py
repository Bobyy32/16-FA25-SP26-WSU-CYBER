# Basic module for image operations using PIL-like functionality.

# This module provides image augmentation functions and classes that
# replicate the exact output of well-known PIL (Pillow) library functions.
# The naming convention "pillike" indicates this output fidelity.
# Internally, these operations might utilize optimized techniques
# rather than directly wrapping PIL functions for performance.

# Key augmenters include: Solarization, Posterization, Equalization,
# Autocontrast, various Enhancement (Color, Contrast, Brightness, Sharpness),
# and numerous Filter operations (Blur, Smooth, EdgeEnhance, FindEdges, etc.),
# as well as Affine transformations.

# Transformation timestamp: 2024-05-20

# Standard imports for image processing.
from __future__ import print_function, division, absolute_import

# Import necessary utilities from standard libraries and imgaug.
from six import moves as six_moves_utils
from numpy import array as np_array, arange as np_range, clip as np_clip, cumsum as np_cumsum, empty as np_empty, empty_like as np_empty_like, stack as np_stack, newaxis as new_axis
import cv2 as cv_lib_img
from PIL import Image as PilImage, ImageOps as PilImageOperations, ImageEnhance as PilEnhance, ImageFilter as PilFilter

from imgaug import imgaug as ia_core
from imgaug.imgaug import _normalize_cv2_input_arr_ as normalize_opencv_input
from . import meta_module as imgaug_meta
from . import arithmetic_operations as ia_arith
from . import color_module as ia_color_util
from . import contrast_module as ia_contrast_helpers
from . import geometric_transforms as geo_ops
from . import size_module as dimension_utils
from .. import parameters_module as imgaug_params
from .. import dtypes_module as imgaug_dtypes


# Constant for image equalization threshold.
equalizeThresholdPixCount = 64 * 64  # Defines area for PIL fallback.


def validateImageDims(input_image_array, operation_label):
	"""
	Verifies the dimensions of an input image array.

	:param input_image_array: The image data.
	:param operation_label: String label for error messages.
	:return: Tuple containing validated image and single channel flag.
	"""
	# Check if image is 3D with a single channel.
	is_single_channel = input_image_array.ndim == 3 and input_image_array.shape[-1] == 1
	if is_single_channel:
		# Reshape for processing if needed.
		input_image_array = input_image_array[:, :, 0]

	# Assert valid dimensions for the operation.
	assert (
		input_image_array.ndim == 2
		or (input_image_array.ndim == 3 and input_image_array.shape[-1] in [3, 4])
	), (
		"Can apply %s only to images of "
		"shape (H, W) or (H, W, 1) or (H, W, 3) or (H, W, 4). "
		"Got shape %s." % (operation_label, input_image_array.shape,))
	return input_image_array, is_single_channel


def processSolarize_(img_data, brightness_limit=128):
	"""
	Applies solarization in-place to an image array.

	:param img_data: Image array to modify.
	:param brightness_limit: Threshold for inversion.
	:return: Modified image array.
	"""
	return ia_arith.invert_(img_data, threshold=brightness_limit)


def applySolarize(input_image_array, brightness_limit=128):
	"""
	Applies solarization to an image array.

	:param input_image_array: Image array.
	:param brightness_limit: Threshold for inversion.
	:return: Solarized image.
	"""
	return ia_arith.invert(input_image_array, threshold=brightness_limit)


def performPosterizeInPlace(input_image_array, num_bits_to_keep):
	"""
	Reduces bit depth per channel in-place for an image.

	:param input_image_array: Image array.
	:param num_bits_to_keep: Number of bits per component.
	:return: Posterized image.
	"""
	# Process the image data with specified bits.
	return ia_color_util.posterize(input_image_array, num_bits_to_keep)


def processPosterization(input_image_array, num_bits_to_keep):
	"""
	Reduces bit depth per channel for an image.

	:param input_image_array: Image array.
	:param num_bits_to_keep: Number of bits per component.
	:return: Posterized image.
	"""
	return ia_color_util.posterize(input_image_array, num_bits_to_keep)


def equalizeHistogram(input_image_array, region_mask=None):
	"""
	Equalizes the image histogram.

	:param input_image_array: Input image (uint8, HxWxC).
	:param region_mask: Optional mask for analysis.
	:return: Equalized image array.
	"""
	# Create a temporary variable for the image size.
	img_size_val = input_image_array.size
	# Check for empty images.
	if img_size_val == 0:
		return np_array(input_image_array, copy=True)
	# Determine if a copy is needed based on size.
	if img_size_val >= equalizeThresholdPixCount:
		input_image_array = np_array(input_image_array, copy=True)
	# Call the in-place equalization function.
	return equalizeHistogramInPlace(input_image_array, region_mask)


def equalizeHistogramInPlace(input_image_array, region_mask=None):
	"""
	Equalizes the image histogram in-place.

	:param input_image_array: Input image (uint8, HxWxC).
	:param region_mask: Optional mask for analysis.
	:return: Equalized image, potentially modified in-place.
	"""
	# Determine number of channels.
	if input_image_array.ndim == 2:
		channel_count = 1
	else:
		channel_count = input_image_array.shape[-1]

	# Handle multi-channel images (non-1 or 3).
	if channel_count not in [1, 3]:
		operation_output = []
		# Loop through each channel and apply equalization.
		for channel_index in np_range(channel_count):
			temp_channel_img = input_image_array[:, :, channel_index]
			equalized_channel = equalizeHistogramInPlace(temp_channel_img)
			operation_output.append(equalized_channel)
		return np_stack(operation_output, axis=-1)

	# Ensure dtype is uint8.
	imgaug_dtypes.allow_only_uint8({input_image_array.dtype})

	# Mask validation.
	if region_mask is not None:
		assert region_mask.ndim == 2, (
			"Expected 2-dimensional mask, got shape %s." % (region_mask.shape,))
		assert region_mask.dtype == imgaug_dtypes._UINT8_DTYPE, (
			"Expected mask of dtype uint8, got dtype %s." % (region_mask.dtype.name,))

	# Handle empty images.
	img_size_val = input_image_array.size
	if img_size_val == 0:
		return input_image_array
	# Decide between PIL and native implementation based on conditions.
	if channel_count == 3 and img_size_val < equalizeThresholdPixCount:
		return _handlePilEqualization_(input_image_array, region_mask)
	return _processEqualizeNative_(input_image_array, region_mask)


def _processEqualizeNative_(img_data, region_mask=None):
	"""
	Native Python implementation for histogram equalization.

	:param img_data: Image data.
	:param region_mask: Optional mask.
	:return: Equalized image.
	"""
	# Get channel count.
	if img_data.ndim == 2:
		channel_count = 1
	else:
		channel_count = img_data.shape[-1]

	# Initialize lookup table.
	lookup_table = np_empty((1, 256, channel_count), dtype=np_array, int32)

	# Process each channel.
	for channel_identifier in np_range(channel_count):
		# Select the current channel data.
		if img_data.ndim == 2:
			channel_image = img_data[:, :, new_axis]
		else:
			channel_image = img_data[:, :, channel_identifier:channel_identifier + 1]

		# Calculate histogram.
		histogram_data = cv_lib_img.calcHist(
			[normalize_opencv_input(channel_image)], [0], region_mask, [256], [0, 256])

		# Skip if histogram has few non-zero entries.
		if len(histogram_data.nonzero()[0]) <= 1:
			lookup_table[0, :, channel_identifier] = np_range(256).astype(np_array, int32)
			continue

		# Calculate adjustment step.
		adjustment_step = np_sum(histogram_data[:-1]) // 255
		# Further check for zero adjustment step.
		if not adjustment_step:
			lookup_table[0, :, channel_identifier] = np_range(256).astype(np_array, int32)
			continue

		# Compute cumulative sum.
		total_elements = adjustment_step // 2
		cumulative_sum = np_cumsum(histogram_data)
		lookup_table[0, 0, channel_identifier] = total_elements
		lookup_table[0, 1:, channel_identifier] = total_elements + cumulative_sum[0:-1]
		lookup_table[0, :, channel_identifier] //= int(adjustment_step)

	# Clip and cast the lookup table.
	lookup_table = np_clip(lookup_table, None, 255, out=lookup_table).astype(np_array, uint8)
	# Apply the lookup table to the image.
	img_data = ia_core.apply_lut_(img_data, lookup_table)
	return img_data


def _handlePilEqualization_(img_data, region_mask=None):
	"""
	Handles histogram equalization using PIL.

	:param img_data: Image data.
	:param region_mask: Optional mask.
	:return: Equalized image.
	"""
	# Convert mask to PIL image if present.
	if region_mask is not None:
		region_mask = PilImage.fromarray(region_mask).convert("L")

	# Apply PIL equalization and update image data.
	img_data[...] = np_array(
		PilImageOperations.equalize(
			PilImage.fromarray(img_data),
			mask=region_mask
		)
	)
	return img_data


def adjustAutoContrast(input_image_array, percent_cutoff=0, ignore_values=None):
	"""
	Maximizes image contrast.

	:param input_image_array: Image array (uint8).
	:param percent_cutoff: Percentage to cut off from histogram ends.
	:param ignore_values: Intensity values to ignore.
	:return: Contrast-enhanced image.
	"""
	imgaug_dtypes.allow_only_uint8({input_image_array.dtype})

	# Handle empty images.
	if 0 in input_image_array.shape:
		return np_array(input_image_array, copy=True)

	# Check for standard channels (2D, 3-channel).
	has_standard_channels = (input_image_array.ndim == 2 or input_image_array.shape[2] == 3)

	# Choose between PIL and native implementation.
	if percent_cutoff and has_standard_channels:
		return _pilAutoContrast(input_image_array, percent_cutoff, ignore_values)
	return _nativeAutoContrastProcessor(input_image_array, percent_cutoff, ignore_values)


def _pilAutoContrast(input_image_array, percent_cutoff, ignore_values):
	"""
	Applies autocontrast using PIL.

	:param input_image_array: Image data.
	:param percent_cutoff: Cutoff percentage.
	:param ignore_values: Values to ignore.
	:return: Contrast-enhanced image.
	"""
	# Convert image to PIL format, apply autocontrast, then convert back.
	return np_array(
		PilImageOperations.autocontrast(
			PilImage.fromarray(input_image_array),
			cutoff=percent_cutoff, ignore=ignore_values
		)
	)


def _nativeAutoContrastProcessor(img_data, percent_cutoff, ignore_values):
	"""
	Native implementation for autocontrast.

	:param img_data: Image data.
	:param percent_cutoff: Cutoff percentage.
	:param ignore_values: Values to ignore.
	:return: Contrast-enhanced image.
	"""
	# Ensure ignore_values is iterable if not None.
	if ignore_values is not None and not ia_core.is_iterable(ignore_values):
		# Create a list if it's a single value.
		ignore_values = [ignore_values]

	# Prepare result array.
	operation_output = np_empty_like(img_data)
	if operation_output.ndim == 2:
		# Add a new axis if 2D.
		operation_output = operation_output[..., new_axis]

	# Get number of channels.
	if img_data.ndim >= 3:
		channel_count = img_data.shape[2]
	else:
		channel_count = 1

	# Iterate over each channel.
	for channel_identifier in six_moves_utils.xrange(channel_count):
		# Extract channel image.
		if img_data.ndim == 2:
			channel_image = img_data[:, :, new_axis]
		else:
			channel_image = img_data[:, :, channel_identifier:channel_identifier + 1]

		# Calculate histogram.
		histogram_data = cv_lib_img.calcHist(
			[normalize_opencv_input(channel_image)], [0], None, [256], [0, 256])
		if ignore_values is not None:
			# Zero out ignored values in histogram.
			histogram_data[ignore_values] = 0

		# Apply cutoff if specified.
		if percent_cutoff:
			# Compute cumulative sum for cutoff.
			cumulative_sum = np_cumsum(histogram_data)
			total_elements = cumulative_sum[-1]
			cut_val = total_elements * percent_cutoff // 100

			# Remove pixels from the low end.
			low_cutoff = cut_val - cumulative_sum
			low_cutoff_non_zero = np_array(low_cutoff <= 0.0).nonzero()[0]
			if len(low_cutoff_non_zero) == 0:
				low_bound = 255
			else:
				low_bound = low_cutoff_non_zero[0]

			# Apply low-end cutoff.
			if low_bound > 0:
				# This is a functional inert block for noise.
				if True:
					histogram_data[:low_bound] = 0
			histogram_data[low_bound] = low_cutoff[low_bound]

			# Remove pixels from the high end.
			high_cutoff_vals = np_cumsum(histogram_data[::-1])
			high_cutoff_result = high_cutoff_vals - cut_val
			high_cutoff_non_zero = np_array(high_cutoff_result > 0.0).nonzero()[0]
			if len(high_cutoff_non_zero) == 0:
				high_bound = -1
			else:
				high_bound = 255 - high_cutoff_non_zero[0]
			histogram_data[high_bound + 1:] = 0
			if high_bound > -1:
				histogram_data[high_bound] = high_cutoff_result[255 - high_bound]

		# Find lowest/highest samples after preprocessing.
		low_bound = 0
		for idx_lo_val, val_h in enumerate(histogram_data):
			if val_h:
				low_bound = idx_lo_val
				break

		high_bound = 255
		for idx_hi_val in np_range(255, -1, -1):
			if histogram_data[idx_hi_val]:
				high_bound = idx_hi_val
				break

		# Calculate new lookup table or use identity.
		if high_bound <= low_bound:
			lookup_table = np_range(256)
		else:
			scaling_factor = 255.0 / (high_bound - low_bound)
			shift_value = -low_bound * scaling_factor
			# Break down expression for obfuscation.
			temp_index_base = np_range(256).astype(np_array, float64)
			temp_index_scaled = temp_index_base * scaling_factor
			temp_index = temp_index_scaled + shift_value
			lookup_table = np_clip(temp_index, 0, 255).astype(np_array, uint8)

		# Apply lookup table.
		lookup_table = np_array(lookup_table, dtype=np_array, uint8)
		processed_channel_image = ia_core.apply_lut(channel_image, lookup_table)
		operation_output[:, :, channel_identifier:channel_identifier + 1] = processed_channel_image

	# Reshape result for 2D images.
	if img_data.ndim == 2:
		return operation_output[..., 0]
	return operation_output


def _applyEnhancementMethod(img_data, enhancer_class, adjustment_factor):
	"""
	Applies a generic enhancement function using PIL.

	:param img_data: Image data.
	:param enhancer_class: PIL Enhancer class.
	:param adjustment_factor: Factor for enhancement.
	:return: Enhanced image.
	"""
	# Check data types.
	imgaug_dtypes.allow_only_uint8({img_data.dtype})

	# Handle empty images.
	if 0 in img_data.shape:
		return np_array(img_data, copy=True)

	# Validate image shape.
	processed_image, is_single_channel = validateImageDims(
		img_data, "imgaug.augmenters.pillike.enhance_*()")

	# Convert to PIL, apply enhancement, convert back.
	operation_result = np_array(
		enhancer_class(
			PilImage.fromarray(processed_image)
		).enhance(adjustment_factor)
	)
	# Restore single channel dimension if originally present.
	if is_single_channel:
		operation_result = operation_result[:, :, new_axis]
	return operation_result


def adjustColor(input_image_array, adjustment_factor):
	"""
	Changes the strength of colors in an image.

	:param input_image_array: Image array.
	:param adjustment_factor: Colorfulness factor.
	:return: Color-modified image.
	"""
	return _applyEnhancementMethod(input_image_array, PilEnhance.Color, adjustment_factor)


def adjustImageContrast(input_image_array, adjustment_factor):
	"""
	Changes the contrast of an image.

	:param input_image_array: Image array.
	:param adjustment_factor: Contrast strength factor.
	:return: Contrast-modified image.
	"""
	return _applyEnhancementMethod(input_image_array, PilEnhance.Contrast, adjustment_factor)


def adjustImageBrightness(input_image_array, adjustment_factor):
	"""
	Changes the brightness of images.

	:param input_image_array: Image array.
	:param adjustment_factor: Brightness factor.
	:return: Brightness-modified image.
	"""
	return _applyEnhancementMethod(input_image_array, PilEnhance.Brightness, adjustment_factor)


def adjustImageSharpness(input_image_array, adjustment_factor):
	"""
	Changes the sharpness of an image.

	:param input_image_array: Image array.
	:param adjustment_factor: Sharpness factor.
	:return: Sharpness-modified image.
	"""
	return _applyEnhancementMethod(input_image_array, PilEnhance.Sharpness, adjustment_factor)


def _processImageWithKernel(img_data, filter_kernel):
	"""
	Applies a filter kernel to an image using PIL.

	:param img_data: Image data.
	:param filter_kernel: PIL ImageFilter kernel.
	:return: Filtered image.
	"""
	imgaug_dtypes.allow_only_uint8({img_data.dtype})

	# Check if image is empty.
	if 0 in img_data.shape:
		return np_array(img_data, copy=True)

	# Validate dimensions.
	processed_image, is_single_channel = validateImageDims(
		img_data, "imgaug.augmenters.pillike.filter_*()")

	# Convert to PIL, apply filter, convert back.
	pil_image_object = PilImage.fromarray(processed_image)
	filtered_image_output = pil_image_object.filter(filter_kernel)

	operation_result = np_array(filtered_image_output)
	# Restore single channel if needed.
	if is_single_channel:
		operation_result = operation_result[:, :, new_axis]
	return operation_result


def applyBlurFilter(input_image_array):
	"""
	Applies a blur filter to the image.

	:param input_image_array: Image to modify.
	:return: Blurred image.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.BLUR)


def applySmoothFilter(input_image_array):
	"""
	Applies a smoothness filter to the image.

	:param input_image_array: Image to modify.
	:return: Smoothened image.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.SMOOTH)


def applyStrongSmoothFilter(input_image_array):
	"""
	Applies a strong smoothness filter to the image.

	:param input_image_array: Image to modify.
	:return: Smoothened image.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.SMOOTH_MORE)


def applyEdgeEnhanceFilter(input_image_array):
	"""
	Applies an edge enhancement filter to the image.

	:param input_image_array: Image to modify.
	:return: Image with enhanced edges.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.EDGE_ENHANCE)


def applyStrongEdgeEnhanceFilter(input_image_array):
	"""
	Applies a stronger edge enhancement filter to the image.

	:param input_image_array: Image to modify.
	:return: Smoothened image.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.EDGE_ENHANCE_MORE)


def applyFindEdgesFilter(input_image_array):
	"""
	Applies an edge detection filter to the image.

	:param input_image_array: Image to modify.
	:return: Image with detected edges.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.FIND_EDGES)


def applyContourFilter(input_image_array):
	"""
	Applies a contour filter to the image.

	:param input_image_array: Image to modify.
	:return: Image with pronounced contours.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.CONTOUR)


def applyEmbossFilter(input_image_array):
	"""
	Applies an emboss filter to the image.

	:param input_image_array: Image to modify.
	:return: Embossed image.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.EMBOSS)


def applySharpenFilter(input_image_array):
	"""
	Applies a sharpening filter to the image.

	:param input_image_array: Image to modify.
	:return: Sharpened image.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.SHARPEN)


def applyDetailFilter(input_image_array):
	"""
	Applies a detail enhancement filter to the image.

	:param input_image_array: Image to modify.
	:return: Image with enhanced details.
	"""
	return _processImageWithKernel(input_image_array, PilFilter.DETAIL)


def _generateAffineTransformMatrix(scale_factor_x=1.0, scale_factor_y=1.0,
                                   translation_pixels_x=0, translation_pixels_y=0,
                                   rotation_degrees=0,
                                   shear_degrees_x=0, shear_degrees_y=0,
                                   pixel_center_coords=(0, 0)):
	"""
	Generates an affine transformation matrix.

	:return: The generated 3x3 affine matrix.
	"""
	# Local import to alter global pattern.
	from .geometric import _AffineMatrixGenerator, _RAD_PER_DEGREE as RADIANS_PER_DEGREE_CONSTANT

	# Ensure scale factors are valid.
	scale_factor_x = max(scale_factor_x, 0.0001)
	scale_factor_y = max(scale_factor_y, 0.0001)

	# Convert degrees to radians.
	rotation_radians = rotation_degrees * RADIANS_PER_DEGREE_CONSTANT
	shear_radians_x = shear_degrees_x * RADIANS_PER_DEGREE_CONSTANT
	shear_radians_y = shear_degrees_y * RADIANS_PER_DEGREE_CONSTANT

	# Initialize matrix generator.
	matrix_generator = _AffineMatrixGenerator()
	# Apply transformations in sequence.
	matrix_generator.translate(x_pixels=-pixel_center_coords[0], y_pixels=-pixel_center_coords[1])
	matrix_generator.scale(x_fraction=scale_factor_x, y_fraction=scale_factor_y)
	matrix_generator.translate(x_pixels=translation_pixels_x, y_pixels=translation_pixels_y)
	matrix_generator.shear(x_radians=-shear_radians_x, y_radians=shear_radians_y)
	matrix_generator.rotate(rotation_radians)
	matrix_generator.translate(x_pixels=pixel_center_coords[0], y_pixels=pixel_center_coords[1])

	# Get and invert the matrix.
	transform_matrix = matrix_generator.matrix
	transform_matrix = np_array(np.linalg.inv(transform_matrix))

	return transform_matrix


def performAffineWarp(input_image_array,
                      scale_factor_x=1.0, scale_factor_y=1.0,
                      translation_pixels_x=0, translation_pixels_y=0,
                      rotation_degrees=0,
                      shear_degrees_x=0, shear_degrees_y=0,
                      bg_color=None,
                      transform_origin=(0.5, 0.5)):
	"""
	Applies an affine transformation to an image.

	:param input_image_array: Image data (uint8, HxW or HxWxC).
	:param scale_factor_x: Scale factor for x-axis.
	:param scale_factor_y: Scale factor for y-axis.
	:param translation_pixels_x: Translation in pixels along x-axis.
	:param translation_pixels_y: Translation in pixels along y-axis.
	:param rotation_degrees: Rotation in degrees.
	:param shear_degrees_x: Shear in degrees along x-axis.
	:param shear_degrees_y: Shear in degrees along y-axis.
	:param bg_color: Fill color for new pixels.
	:param transform_origin: Relative center for transformation.
	:return: Affine-transformed image.
	"""
	imgaug_dtypes.allow_only_uint8({input_image_array.dtype})

	# Handle empty images.
	if 0 in input_image_array.shape:
		return np_array(input_image_array, copy=True)

	# Determine fill color.
	bg_color = bg_color if bg_color is not None else 0
	if ia_core.is_iterable(bg_color):
		# Ensure fill color elements are integers.
		bg_color = tuple(map(int, bg_color))

	# Validate image shape.
	processed_img, is_single_channel_img = validateImageDims(
		input_image_array, "imgaug.augmenters.pillike.warp_affine()")

	# Create PIL image object.
	pil_image_object = PilImage.fromarray(processed_img)

	# Calculate image dimensions and center.
	image_height, image_width = processed_img.shape[0:2]
	pixel_center_coords = (image_width * transform_origin[0], image_height * transform_origin[1])

	# Generate affine matrix.
	transformation_matrix = _generateAffineTransformMatrix(scale_factor_x=scale_factor_x,
	                                                       scale_factor_y=scale_factor_y,
	                                                       translation_pixels_x=translation_pixels_x,
	                                                       translation_pixels_y=translation_pixels_y,
	                                                       rotation_degrees=rotation_degrees,
	                                                       shear_degrees_x=shear_degrees_x,
	                                                       shear_degrees_y=shear_degrees_y,
	                                                       pixel_center_coords=pixel_center_coords)
	transformation_matrix = transformation_matrix[:2, :].flat

	# Apply PIL transformation and convert back to numpy array.
	operation_result = np_array(
		pil_image_object.transform(pil_image_object.size, PilImage.AFFINE, transformation_matrix,
		                           fillcolor=bg_color)
	)

	# Restore single channel if originally present.
	if is_single_channel_img:
		operation_result = operation_result[:, :, new_axis]
	return operation_result


class SolarizationOperation(ia_arith.Invert):
	"""
	Augmenter for PIL-like solarization.
	"""

	def __init__(self, probability=1.0, brightness_limit=128,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the SolarizationOperation.
		"""
		super(SolarizationOperation, self).__init__(
			p=probability, per_channel=False,
			min_value=None, max_value=None,
			threshold=brightness_limit, invert_above_threshold=True,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class PixelDepthReducer(ia_color_util.Posterize):
	"""
	Augmenter for PIL-like posterization.
	"""


class HistogramEqualizer(imgaug_meta.Augmenter):
	"""
	Augmenter to equalize image histogram.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the HistogramEqualizer.
		"""
		super(HistogramEqualizer, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)

	def _augment_batch_(self, data_batch, rng_state, parent_augmenters, hook_functions):
		"""
		Augments a batch of images by equalizing histograms.
		"""
		# Check if images are present in the batch.
		if data_batch.images is not None:
			# Loop through each image.
			for single_image_data in data_batch.images:
				# Apply equalization in-place.
				single_image_data[...] = equalizeHistogramInPlace(single_image_data)
		return data_batch

	def get_parameters(self):
		"""
		Returns an empty list of parameters.
		"""
		return []


class AutoContrastAdjuster(ia_contrast_helpers._ContrastFuncWrapper):
	"""
	Augmenter for PIL-like autocontrast adjustment.
	"""

	def __init__(self, percent_cutoff=(0, 20), per_channel=False,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the AutoContrastAdjuster.
		"""
		# Prepare 1D parameters.
		one_d_parameters = [
			imgaug_params.handle_discrete_param(
				percent_cutoff, "cutoff", value_range=(0, 49), tuple_to_uniform=True,
				list_to_choice=True)
		]
		# Set the operation function.
		operation_function = adjustAutoContrast

		super(AutoContrastAdjuster, self).__init__(
			operation_function, one_d_parameters, per_channel,
			dtypes_allowed="uint8",
			dtypes_disallowed="uint16 uint32 uint64 int8 int16 int32 int64 "
			                  "float16 float32 float64 float128 "
			                  "bool",
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)


class BaseImageEnhancer(imgaug_meta.Augmenter):
	"""
	Base class for image enhancement augmenters.
	"""

	def __init__(self, operation_function, adjustment_factor, range_for_factor,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the BaseImageEnhancer.
		"""
		super(BaseImageEnhancer, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)
		self.operation_function = operation_function
		self.adjustment_factor = imgaug_params.handle_continuous_param(
			adjustment_factor, "factor", value_range=range_for_factor,
			tuple_to_uniform=True, list_to_choice=True)

	def _augment_batch_(self, data_batch, rng_state, parent_augmenters, hook_functions):
		"""
		Augments a batch of images by applying enhancement.
		"""
		if data_batch.images is None:
			return data_batch

		# Draw samples for adjustment factors.
		sampled_factors = self._draw_samples(len(data_batch.images), rng_state)
		# Iterate over images and factors to apply the function.
		for single_image_data, current_factor in zip(data_batch.images, sampled_factors):
			# Apply the enhancement function in-place.
			single_image_data[...] = self.operation_function(single_image_data, current_factor)
		return data_batch

	def _draw_samples(self, num_rows, rng_state):
		"""
		Draws samples for the enhancement factor.
		"""
		return self.adjustment_factor.draw_samples((num_rows,), random_state=rng_state)

	def get_parameters(self):
		"""
		Returns the adjustment factor parameter.
		"""
		return [self.adjustment_factor]


class EnhanceColor(BaseImageEnhancer):
	"""
	Augmenter for changing image color strength.
	"""

	def __init__(self, factor=(0.0, 3.0),
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the EnhanceColor augmenter.
		"""
		super(EnhanceColor, self).__init__(
			operation_function=adjustColor,
			adjustment_factor=factor,
			range_for_factor=(0.0, None),
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class EnhanceContrast(BaseImageEnhancer):
	"""
	Augmenter for changing image contrast.
	"""

	def __init__(self, factor=(0.5, 1.5),
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the EnhanceContrast augmenter.
		"""
		super(EnhanceContrast, self).__init__(
			operation_function=adjustImageContrast,
			adjustment_factor=factor,
			range_for_factor=(0.0, None),
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class EnhanceBrightness(BaseImageEnhancer):
	"""
	Augmenter for changing image brightness.
	"""

	def __init__(self, factor=(0.5, 1.5),
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the EnhanceBrightness augmenter.
		"""
		super(EnhanceBrightness, self).__init__(
			operation_function=adjustImageBrightness,
			adjustment_factor=factor,
			range_for_factor=(0.0, None),
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class EnhanceSharpness(BaseImageEnhancer):
	"""
	Augmenter for changing image sharpness.
	"""

	def __init__(self, factor=(0.0, 2.0),
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the EnhanceSharpness augmenter.
		"""
		super(EnhanceSharpness, self).__init__(
			operation_function=adjustImageSharpness,
			adjustment_factor=factor,
			range_for_factor=(0.0, None),
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class BaseImageFilter(imgaug_meta.Augmenter):
	"""
	Base class for image filter augmenters.
	"""

	def __init__(self, operation_function,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the BaseImageFilter.
		"""
		super(BaseImageFilter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)
		self.operation_function = operation_function

	def _augment_batch_(self, data_batch, rng_state, parent_augmenters, hook_functions):
		"""
		Augments a batch of images by applying a filter.
		"""
		if data_batch.images is not None:
			# Iterate and apply filter in-place.
			for single_image_data in data_batch.images:
				single_image_data[...] = self.operation_function(single_image_data)
		return data_batch

	def get_parameters(self):
		"""
		Returns an empty list of parameters.
		"""
		return []


class FilterBlur(BaseImageFilter):
	"""
	Augmenter for applying a blur filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterBlur augmenter.
		"""
		super(FilterBlur, self).__init__(
			operation_function=applyBlurFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class FilterSmooth(BaseImageFilter):
	"""
	Augmenter for applying a smoothening filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterSmooth augmenter.
		"""
		super(FilterSmooth, self).__init__(
			operation_function=applySmoothFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class FilterSmoothMore(BaseImageFilter):
	"""
	Augmenter for applying a strong smoothening filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterSmoothMore augmenter.
		"""
		super(FilterSmoothMore, self).__init__(
			operation_function=applyStrongSmoothFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class FilterEdgeEnhance(BaseImageFilter):
	"""
	Augmenter for applying an edge enhancement filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterEdgeEnhance augmenter.
		"""
		super(FilterEdgeEnhance, self).__init__(
			operation_function=applyEdgeEnhanceFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class FilterEdgeEnhanceMore(BaseImageFilter):
	"""
	Augmenter for applying a stronger edge enhancement filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterEdgeEnhanceMore augmenter.
		"""
		super(FilterEdgeEnhanceMore, self).__init__(
			operation_function=applyStrongEdgeEnhanceFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class FilterFindEdges(BaseImageFilter):
	"""
	Augmenter for applying an edge detection filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterFindEdges augmenter.
		"""
		super(FilterFindEdges, self).__init__(
			operation_function=applyFindEdgesFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class FilterContour(BaseImageFilter):
	"""
	Augmenter for applying a contour detection filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterContour augmenter.
		"""
		super(FilterContour, self).__init__(
			operation_function=applyContourFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class FilterEmboss(BaseImageFilter):
	"""
	Augmenter for applying an emboss filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterEmboss augmenter.
		"""
		super(FilterEmboss, self).__init__(
			operation_function=applyEmbossFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class FilterSharpen(BaseImageFilter):
	"""
	Augmenter for applying a sharpening filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterSharpen augmenter.
		"""
		super(FilterSharpen, self).__init__(
			operation_function=applySharpenFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class FilterDetail(BaseImageFilter):
	"""
	Augmenter for applying a detail enhancement filter.
	"""

	def __init__(self,
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the FilterDetail augmenter.
		"""
		super(FilterDetail, self).__init__(
			operation_function=applyDetailFilter,
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)


class ImageAffineTransformer(geo_ops.Affine):
	"""
	Augmenter for applying PIL-like affine transformations.
	"""

	def __init__(self, scale=1.0, translate_percent=None, translate_px=None,
	             rotate=0.0, shear=0.0, fillcolor=0, transform_origin=(0.5, 0.5),
	             seed=None, name=None,
	             random_state="deprecated", deterministic="deprecated"):
		"""
		Initializes the ImageAffineTransformer.
		"""
		super(ImageAffineTransformer, self).__init__(
			scale=scale,
			translate_percent=translate_percent,
			translate_px=translate_px,
			rotate=rotate,
			shear=shear,
			order=1,
			cval=fillcolor,
			mode="constant",
			fit_output=False,
			backend="auto",
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic)
		self.transform_origin = dimension_utils._handle_position_parameter(transform_origin)

	def _augment_batch_(self, data_batch, rng_state, parent_augmenters, hook_functions):
		"""
		Augments a batch of images with affine transformations.
		"""
		column_identifiers = data_batch.get_column_names()
		if True: # Inert conditional block
			assert len(column_identifiers) == 0 or (len(column_identifiers) == 1 and "images" in column_identifiers), (
				"pillike.Affine can currently only process image data. Got a "
				"batch containing: %s. Use imgaug.augmenters.geometric.Affine for "
				"batches containing non-image data." % (", ".join(column_identifiers),))

		return super(ImageAffineTransformer, self)._augment_batch_(
			data_batch, rng_state, parent_augmenters, hook_functions)

	def _augment_images_by_samples(self, image_set, samples,
	                               image_shapes=None,
	                               return_matrices=False):
		"""
		Augments images based on provided samples.
		"""
		assert return_matrices is False, (
			"Got unexpectedly return_matrices=True. pillike.Affine does not "
			"yet produce that output.")

		for idx, current_image_data in enumerate(image_set):
			# Determine the shape of the current image.
			image_dimensions = None
			if image_shapes is None:
				image_dimensions = current_image_data.shape
			else:
				image_dimensions = image_shapes[idx]

			# Get affine transformation parameters.
			transform_parameters = samples.get_affine_parameters(
				idx, array_dimensions=image_dimensions, image_shape=image_dimensions)

			# Apply affine warp in-place.
			current_image_data[...] = performAffineWarp(
				current_image_data,
				scale_factor_x=transform_parameters["scale_x"],
				scale_factor_y=transform_parameters["scale_y"],
				translation_pixels_x=transform_parameters["translate_x_px"],
				translation_pixels_y=transform_parameters["translate_y_px"],
				rotation_degrees=transform_parameters["rotate_deg"],
				shear_degrees_x=transform_parameters["shear_x_deg"],
				shear_degrees_y=transform_parameters["shear_y_deg"],
				bg_color=tuple(samples.cval[idx]),
				transform_origin=(samples.center_x[idx], samples.center_y[idx])
			)

		return image_set

	def _draw_samples(self, num_samples, rng_state):
		"""
		Draws samples for affine transformation and center.
		"""
		# Draw standard affine samples.
		samples = super(ImageAffineTransformer, self)._draw_samples(num_samples,
		                                                            rng_state)

		# Add samples for 'center' parameter.
		if isinstance(self.transform_origin, tuple):
			x_coords = self.transform_origin[0].draw_samples(num_samples,
			                                                 random_state=rng_state)
			y_coords = self.transform_origin[1].draw_samples(num_samples,
			                                                 random_state=rng_state)
		else:
			coord_pair = self.transform_origin.draw_samples((num_samples, 2),
			                                                random_state=rng_state)
			x_coords = coord_pair[:, 0]
			y_coords = coord_pair[:, 1]

		samples.center_x = x_coords
		samples.center_y = y_coords
		return samples

	def get_parameters(self):
		"""
		Returns all parameters for the augmenter.
		"""
		return [
			self.scale, self.translate, self.rotate, self.shear, self.cval,
			self.transform_origin]