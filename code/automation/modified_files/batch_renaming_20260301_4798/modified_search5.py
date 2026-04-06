from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from nni.algorithms.nas.tensorflow import enas
import datasets
from macro import GeneralNetwork
from micro import MicroNetwork
from utils import accuracy, accuracy_metrics

a = datasets.get_dataset()
b = MicroNetwork()
c = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
d = SGD(learning_rate=0.05, momentum=0.9)
e = enas.EnasTrainer(b, loss=c, metrics=accuracy_metrics, reward_function=accuracy, optimizer=d, batch_size=64, num_epochs=310, dataset_train=a[0], dataset_valid=a[1])
e.train()