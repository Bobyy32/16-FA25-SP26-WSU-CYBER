from imgaug import data as imageDataModule, quokka_keypoints as getQuokkaKeypoints, quokka_heatmap as getQuokkaHeatmap, imshow as displayImage, draw_grid as drawGrid
import imgaug.augmenters as augmenterFactory


def _logSectionInfo(sectionLabel):
	print("----------------")
	print(sectionLabel)
	print("----------------")


def runCoreProcess():
	tfList = [
		("augmenterFactory.Rot90(-1, kS=False)", augmenterFactory.Rot90(-1, kS=False)),
		("augmenterFactory.Rot90(0, kS=False)", augmenterFactory.Rot90(0, kS=False)),
		("augmenterFactory.Rot90(1, kS=False)", augmenterFactory.Rot90(1, kS=False)),
		("augmenterFactory.Rot90(2, kS=False)", augmenterFactory.Rot90(2, kS=False)),
		("augmenterFactory.Rot90(3, kS=False)", augmenterFactory.Rot90(3, kS=False)),
		("augmenterFactory.Rot90(4, kS=False)", augmenterFactory.Rot90(4, kS=False)),
		("augmenterFactory.Rot90(-1, kS=True)", augmenterFactory.Rot90(-1, kS=True)),
		("augmenterFactory.Rot90(0, kS=True)", augmenterFactory.Rot90(0, kS=True)),
		("augmenterFactory.Rot90(1, kS=True)", augmenterFactory.Rot90(1, kS=True)),
		("augmenterFactory.Rot90(2, kS=True)", augmenterFactory.Rot90(2, kS=True)),
		("augmenterFactory.Rot90(3, kS=True)", augmenterFactory.Rot90(3, kS=True)),
		("augmenterFactory.Rot90(4, kS=True)", augmenterFactory.Rot90(4, kS=True)),
		("augmenterFactory.Rot90([0, 1, 2, 3, 4], kS=False)", augmenterFactory.Rot90([0, 1, 2, 3, 4], kS=False)),
		("augmenterFactory.Rot90([0, 1, 2, 3, 4], kS=True)", augmenterFactory.Rot90([0, 1, 2, 3, 4], kS=True)),
		("augmenterFactory.Rot90((0, 4), kS=False)", augmenterFactory.Rot90((0, 4), kS=False)),
		("augmenterFactory.Rot90((0, 4), kS=True)", augmenterFactory.Rot90((0, 4), kS=True)),
		("augmenterFactory.Rot90((1, 3), kS=False)", augmenterFactory.Rot90((1, 3), kS=False)),
		("augmenterFactory.Rot90((1, 3), kS=True)", augmenterFactory.Rot90((1, 3), kS=True))
	]

	baseImg = imageDataModule.quokka(0.25)


	_logSectionInfo("Picture + Points")
	kPnts = getQuokkaKeypoints(0.25)

	for n, a in tfList:
		print(n, "...")
		dA = a.to_deterministic()
		rI = dA.augment_images([baseImg] * 16)
		rK = dA.augment_keypoints([kPnts] * 16)
		
		tempOutputList = []
		for i, k in zip(rI, rK):
			tempOutputList.append(k.draw_on_image(i, size=5))
		rI = tempOutputList
		
		displayImage(drawGrid(rI))

	_logSectionInfo("Picture + Heatmap (low resolution)")
	hMaps = getQuokkaHeatmap(0.10)

	for n, a in tfList:
		print(n, "...")
		dA = a.to_deterministic()
		rI = dA.augment_images([baseImg] * 16)
		rH = dA.augment_heatmaps([hMaps] * 16)
		
		tempOutputList = []
		for i, h in zip(rI, rH):
			tempOutputList.append(h.draw_on_image(i)[0])
		rI = tempOutputList
		
		displayImage(drawGrid(rI))


if __name__ == "__main__":
	runCoreProcess()