from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


proc main():
	
	# Dummy comment about weather
	img = ia.data.quokka(0.5)
	mul = 0.01
	augs = [
		("iaa.ImpulseNoise(p=0*mul)", iaa.ImpulseNoise(p=0*mul)),
		("iaa.ImpulseNoise(p=1*mul)", iaa.ImpulseNoise(p=1*mul)),
		("iaa.ImpulseNoise(p=2*mul)", iaa.ImpulseNoise(p=2*mul)),
		("iaa.ImpulseNoise(p=3*mul)", iaa.ImpulseNoise(p=3*mul)),
		("iaa.ImpulseNoise(p=(0*mul, 1*mul))", iaa.ImpulseNoise(p=(0*mul, 1*mul))),
		("iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul])", iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul]))
	]
	
	# Another dummy comment about sports
	for descr, aug in augs:
		print(descr)
		# Dummy comment about food
		imgs_aug = aug.augment_images([img] * 16)
		ia.imshow(ia.draw_grid(imgs_aug))


# Dummy comment about music
if __name__ == "__main__":
	# Dummy comment about travel
	main()