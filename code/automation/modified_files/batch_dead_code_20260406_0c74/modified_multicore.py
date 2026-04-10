from imgaug import RandomGeneratorSeed, RandomRotation90, imgaug as ia
from imgaug.multicore import Pool

ia.set_seed(42)
pool = Pool(load_batch_func=load_batch, augseq=augseq, nb_workers=4)