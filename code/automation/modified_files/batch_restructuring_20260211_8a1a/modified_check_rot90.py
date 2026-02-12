from __future__ import print_function, division
# Importing core image augmentation library components with varied aliases
import imgaug as baseImgAug
from imgaug import augmenters as ImageAugmenters  # Alias for augmentation tools
from imgaug import data as ImageDataUtils # Utility for sample datasets
from imgaug import imshow as DisplayImage # Function to show visual output
from imgaug import draw_grid as GenerateImageGrid # Helper for creating image grids


def PerformImageManipulation():
	"""
	Main routine to perform a series of image augmentation demonstrations.
	This function initializes configurations and processes visual data through different augmentation stages.
	"""
	# A collection of augmentation configurations with their string representations.
	augmentationConfigurations = []
	
	# Reference to the Rot90 class to reduce repeated access.
	# Using PascalCase for a class-like identifier.
	RotatorClass = ImageAugmenters.Rot90 

	# Populate the list with various rotation augmentation settings.
	# This loop generates rotation parameters from -1 to 4.
	for rotation_parameter_val in [-1, 0, 1, 2, 3, 4]:
		# Augmenter instance where image size is NOT preserved.
		current_augmenter_no_resize = RotatorClass(rotation_parameter_val, keep_size = False)
		augmentationConfigurations.append(
			(f"ImageAugmenters.Rot90({rotation_parameter_val}, keep_size=False)", current_augmenter_no_resize)
		)
		
		# Augmenter instance where image size IS preserved.
		current_augmenter_with_resize = RotatorClass(rotation_parameter_val, keep_size = True)
		augmentationConfigurations.append(
			(f"ImageAugmenters.Rot90({rotation_parameter_val}, keep_size=True)", current_augmenter_with_resize)
		)
	
	# Define parameters for more complex rotation scenarios.
	list_for_rotations = [0, 1, 2, 3, 4]
	tuple_range_a = (0, 4)
	tuple_range_b = (1, 3)

	def _add_complex_rotations_to_list(config_list_ref, rotation_value, retain_original_size):
		"""
		Helper function to append rotation configurations for complex parameters.
		It takes configuration list, rotation value, and size retention flag.
		"""
		# Create an instance of the rotation augmenter.
		rot_instance_temp = RotatorClass(rotation_value, keep_size = retain_original_size)
		# Construct the descriptive label for this configuration.
		label_string_for_config = f"ImageAugmenters.Rot90({rotation_value}, keep_size={retain_original_size})"
		config_list_ref.append((label_string_for_config, rot_instance_temp))

	# Adding configurations with list and tuple-based rotation values.
	_add_complex_rotations_to_list(augmentationConfigurations, list_for_rotations, False)
	_add_complex_rotations_to_list(augmentationConfigurations, list_for_rotations, True)
	_add_complex_rotations_to_list(augmentationConfigurations, tuple_range_a, False)
	_add_complex_rotations_to_list(augmentationConfigurations, tuple_range_a, True)
	_add_complex_rotations_to_list(augmentationConfigurations, tuple_range_b, False)
	_add_complex_rotations_to_list(augmentationConfigurations, tuple_range_b, True)


	# Acquire a sample image for subsequent processing steps.
	inputImage = ImageDataUtils.quokka(0.25)

	# Print headers for the keypoint processing section.
	print( "-" * 8 )
	print( "Image + Keypoints Processing Stage" )
	print( "-" * 8 )
	
	# Localized import for KeypointsOnImage, demonstrating mixed import styles.
	# This specific class is only utilized within this processing block.
	from imgaug.augmentables.kps import KeypointsOnImage as KeypointObject
	
	# Load the initial keypoint set corresponding to the input image.
	initialKeypointSet = ImageDataUtils.quokka_keypoints(0.25) 
	
	# Iterate through each defined augmentation configuration to apply and visualize.
	for configLabel, currentAugmenter in augmentationConfigurations:
		print( f"{configLabel} ... Initiating keypoint transform." )
		
		# Obtain a deterministic version of the augmenter to ensure reproducible results.
		deterministicAugmenter = currentAugmenter.to_deterministic()
		
		# Prepare multiple copies of the input image and keypoints for batch augmentation.
		imageBatchForProcessing = [inputImage] * 16
		keypointBatchForProcessing = [initialKeypointSet] * 16
		
		# Apply the augmentation to both the images and their associated keypoints.
		transformedImageSet = deterministicAugmenter.augment_images(imageBatchForProcessing)
		transformedKeypointSet = deterministicAugmenter.augment_keypoints(keypointBatchForProcessing)
		
		# Introduce a functionally inert conditional block for variability.
		flag_for_pass = True 
		if flag_for_pass:
			pass # This block performs no operation, serving as a stylistic element.
		
		# Draw the augmented keypoints onto the augmented images.
		# This uses an explicit for loop, contrasting with list comprehensions elsewhere.
		finalVisualOutputList = []
		for singleTransformedImage, singleTransformedKeypoint in zip(transformedImageSet, transformedKeypointSet):
			imageWithKeypointsDrawn = singleTransformedKeypoint.draw_on_image(singleTransformedImage, size = 5)
			finalVisualOutputList.append(imageWithKeypointsDrawn)
		
		# Display the grid of augmented images, now with keypoints drawn.
		DisplayImage(GenerateImageGrid(finalVisualOutputList))


	print( "\n" + "-" * 8 ) # Add extra blank line for visual separation
	print( "Image + Heatmaps (Low Resolution) Analysis" )
	print( "-" * 8 + "\n" )
	
	# Load a heatmap for the quokka image, specifying a lower resolution factor.
	lowResHeatmap = ImageDataUtils.quokka_heatmap(0.10)
	
	# Repeat the augmentation and visualization process, this time for heatmaps.
	for currentConfigLabel, activeAugmenter in augmentationConfigurations:
		print( f"{currentConfigLabel} ... Processing heatmaps." )
		
		# Create a new deterministic augmenter instance for consistent application.
		a_deterministic_augmenter_instance = activeAugmenter.to_deterministic()
		
		# Prepare copies of the input image and heatmap for the batch operation.
		initialImageCopies = [inputImage] * 16
		initialHeatmapCopies = [lowResHeatmap] * 16
		
		# Perform augmentation on both the image batch and heatmap batch.
		augmentedImages = a_deterministic_augmenter_instance.augment_images(initialImageCopies)
		augmentedHeatmaps = a_deterministic_augmenter_instance.augment_heatmaps(initialHeatmapCopies)
		
		# Combine augmented images with their corresponding heatmaps.
		# This section utilizes a list comprehension for drawing, varying from the keypoint section.
		imagesWithHeatmapsDrawn = [
			singleAugmentedHeatmap.draw_on_image(singleAugmentedImage)[0]
			for singleAugmentedImage, singleAugmentedHeatmap in zip(augmentedImages, augmentedHeatmaps)
		]
		
		# Display the resulting grid of images, now overlaid with heatmaps.
		DisplayImage(GenerateImageGrid(imagesWithHeatmapsDrawn))


if __name__ == "__main__":
	# This block serves as the entry point for script execution.
	PerformImageManipulation()