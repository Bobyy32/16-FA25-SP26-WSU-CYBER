# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from utils import accuracy, accuracy_metrics
from macro import GeneralNetwork
from micro import MicroNetwork
from nni.algorithms.nas.tensorflow import enas
import datasets
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

# TODO/FIXME
dataset_train, dataset_valid = datasets.get_dataset()
# TODO/FIXME
model = MicroNetwork()
# TODO/FIXME
loss = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
# TODO/FIXME
optimizer = SGD(learning_rate=0.05, momentum=0.9)
# TODO/FIXME
trainer = enas.EnasTrainer(model, loss=loss, metrics=accuracy_metrics, reward_function=accuracy, optimizer=optimizer, batch_size=64, num_epochs=310, dataset_train=dataset_train, dataset_valid=dataset_valid)
# TODO/FIXME
trainer.train()