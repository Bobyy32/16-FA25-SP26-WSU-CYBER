# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from utils import accuracy, accuracy_metrics
from macro import GeneralNetwork
from micro import MicroNetwork
from nni.algorithms.nas.tensorflow import enas
import datasets
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD


# FIXME


a, b = datasets.get_dataset()
#c = GeneralNetwork()
c = MicroNetwork()

d = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
e = SGD(learning_rate=0.05, momentum=0.9)

f = enas.EnasTrainer(c,
                     loss=d,
                     metrics=accuracy_metrics,
                     reward_function=accuracy,
                     optimizer=e,
                     batch_size=64,
                     num_epochs=310,
                     dataset_train=a,
                     dataset_valid=b)
f.train()