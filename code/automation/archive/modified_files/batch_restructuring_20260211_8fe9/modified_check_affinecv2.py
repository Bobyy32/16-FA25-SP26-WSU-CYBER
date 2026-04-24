from __future__ import print_function, division
import imgaug
from imgaug import (
	Keypoint, KeypointsOnImage, BoundingBox, BoundingBoxesOnImage,
	imresize_single_image, ALL, imshow
)
import imgaug.augmenters as img_aug_mod
import imageio as img_io_ops
import numpy as num_py_mod
import skimage.data as sk_data_mod
import cv2 as cv_ops

numRows = 10
numCols = 10

imgHeight = 200
imgWidth = 256

boxX1 = 64
boxX2 = imgWidth - 64
boxY1 = 64
boxY2 = imgHeight - 64


def generatePoints(rC, cC, bX1, bY1, bX2, bY2):
	pnts = []
	# Calculate step increments
	yStep = (bY2 - bY1) / (cC - 1)
	xStep = (bX2 - bX1) / (rC - 1)

	# Populate point list via iteration
	for rIdx in range(rC):
		cY = bY1 + int(rIdx * yStep)
		for cIdx in range(cC):
			cX = bX1 + int(cIdx * xStep)
			pnts.append((cX, cY))
	# Ensure uniqueness of collected points
	return set(pnts)


def executeProcessing():
	imgAsset = sk_data_mod.astronaut()
	imgAsset = imresize_single_image(imgAsset, (imgHeight, imgWidth))

	# Construct keypoints from generated data
	tempPoints = generatePoints(numRows, numCols, boxX1, boxY1, boxX2, boxY2)
	keyPoints = []
	for ptPair in tempPoints:
		aX, aY = ptPair
		kP = Keypoint(x=aX, y=aY)
		keyPoints.append(kP)
	keyPoints = KeypointsOnImage(keyPoints, shape=imgAsset.shape)

	# Define bounding box structures
	boundBox = BoundingBox(x1=boxX1, x2=boxX2, y1=boxY1, y2=boxY2)
	boundBoxes = BoundingBoxesOnImage([boundBox], shape=imgAsset.shape)

	resultImages = []
	transformSequences = [
		img_aug_mod.AffineCv2(rotate=45),
		img_aug_mod.AffineCv2(translate_px=20),
		img_aug_mod.AffineCv2(translate_percent=0.1),
		img_aug_mod.AffineCv2(scale=1.2),
		img_aug_mod.AffineCv2(scale=0.8),
		img_aug_mod.AffineCv2(shear=45),
		img_aug_mod.AffineCv2(rotate=45, cval=256),
		img_aug_mod.AffineCv2(
			translate_px=20, mode=cv_ops.BORDER_CONSTANT
		),
		img_aug_mod.AffineCv2(
			translate_px=20, mode=cv_ops.BORDER_REPLICATE
		),
		img_aug_mod.AffineCv2(
			translate_px=20, mode=cv_ops.BORDER_REFLECT
		),
		img_aug_mod.AffineCv2(
			translate_px=20, mode=cv_ops.BORDER_REFLECT_101
		),
		img_aug_mod.AffineCv2(
			translate_px=20, mode=cv_ops.BORDER_WRAP
		),
		img_aug_mod.AffineCv2(translate_px=20, mode="constant"),
		img_aug_mod.AffineCv2(translate_px=20, mode="replicate"),
		img_aug_mod.AffineCv2(translate_px=20, mode="reflect"),
		img_aug_mod.AffineCv2(translate_px=20, mode="reflect_101"),
		img_aug_mod.AffineCv2(translate_px=20, mode="wrap"),
		img_aug_mod.AffineCv2(scale=0.5, order=cv_ops.INTER_NEAREST),
		img_aug_mod.AffineCv2(scale=0.5, order=cv_ops.INTER_LINEAR),
		img_aug_mod.AffineCv2(scale=0.5, order=cv_ops.INTER_CUBIC),
		img_aug_mod.AffineCv2(scale=0.5, order=cv_ops.INTER_LANCZOS4),
		img_aug_mod.AffineCv2(scale=0.5, order="nearest"),
		img_aug_mod.AffineCv2(scale=0.5, order="linear"),
		img_aug_mod.AffineCv2(scale=0.5, order="cubic"),
		img_aug_mod.AffineCv2(scale=0.5, order="lanczos4"),
		img_aug_mod.AffineCv2(rotate=45, translate_px=20, scale=1.2),
		img_aug_mod.AffineCv2(rotate=45, translate_px=20, scale=0.8),
		img_aug_mod.AffineCv2(
			rotate=(-45, 45), translate_px=(-20, 20),
			scale=(0.8, 1.2), order=ALL, mode=ALL, cval=ALL
		),
		img_aug_mod.AffineCv2(
			rotate=(-45, 45), translate_px=(-20, 20),
			scale=(0.8, 1.2), order=ALL, mode=ALL, cval=ALL
		),
		img_aug_mod.AffineCv2(
			rotate=(-45, 45), translate_px=(-20, 20),
			scale=(0.8, 1.2), order=ALL, mode=ALL, cval=ALL
		),
		img_aug_mod.AffineCv2(
			rotate=(-45, 45), translate_px=(-20, 20),
			scale=(0.8, 1.2), order=ALL, mode=ALL, cval=ALL
		)
	]

	for currentSequence in transformSequences:
		deterministicSeq = currentSequence.to_deterministic()

		# Apply augmentation to components
		augmentedKeyPoints = deterministicSeq.augment_keypoints([keyPoints])[0]
		augmentedBoundBoxes = deterministicSeq.augment_bounding_boxes([boundBoxes])[0]
		augmentedImage = deterministicSeq.augment_image(imgAsset)

		# Render initial state for comparison
		initialImage = num_py_mod.copy(imgAsset)
		initialImage = keyPoints.draw_on_image(initialImage)
		initialImage = boundBoxes.draw_on_image(initialImage)

		# Render transformed state
		transformedImage = num_py_mod.copy(augmentedImage)
		transformedImage = augmentedKeyPoints.draw_on_image(transformedImage)
		transformedImage = augmentedBoundBoxes.draw_on_image(transformedImage)

		# Combine for display
		combinedPair = num_py_mod.hstack((initialImage, transformedImage))
		resultImages.append(combinedPair)

	# Final display and persistence operations
	imshow(num_py_mod.vstack(resultImages))
	img_io_ops.imwrite("affinecv2.jpg", num_py_mod.vstack(resultImages))


if __name__ == "__main__":
	executeProcessing()