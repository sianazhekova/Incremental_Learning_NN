# Mini-class of Cifar10
from tensorflow.keras import datasets
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from datasets.Dataset import Dataset
from datasets.Cifar10 import CIFAR10

class MiniCifar10(CIFAR10):

    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    X_train, y_train = CIFAR10.X_train[:2000], CIFAR10.y_train[:2000]
    # X_test, y_test = CIFAR10.X_test[:200], CIFAR10.y_test[:200]  