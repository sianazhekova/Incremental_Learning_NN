import tensorflow as tf
import numpy as np
import copy
from tensorflow.keras import datasets, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CIFAR10(Dataset):
    _all_class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

    _dataset_name = "Cifar 10"
    
    _height = 32
    _width = 32
    _num_channels = 3
    
    _batch_size = 256 #args.batch_size
    
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    
    @staticmethod
    def scale_pixels(dataset):
        return dataset.astype('float32')/255.0
    
    @classmethod
    def custom_dataset_filter(cls, labels_to_keep):
        """ Filter from the underlying dataset images and labels according using the specified labels argument  """
        data, labels = cls.filter_dataset(cls.X_train, cls.y_train, labels_to_keep)
        return data, labels

    @classmethod
    def filter_dataset(cls, data, labels, labels_to_keep):
        """Filter a dataset to contain only data with the specified labels."""
        data = cls.scale_pixels(data)

        # Filter data.
        keep_indices = np.isin(labels, labels_to_keep)
        filtered_data = data[keep_indices[:, 0]]
        filtered_labels = labels[keep_indices]
        filtered_labels = to_categorical(filtered_labels, num_classes=cls.get_default_num_classes())

        return filtered_data, filtered_labels
    
    @classmethod
    def get_iterators(cls, labels_to_keep):
        """Get training and validation data iterators, which have been filtered
        to only contain data matching the given labels."""
        data, labels = cls.filter_dataset(cls.X_train, cls.y_train, labels_to_keep)

        # Training/validation splits.
        # Use separate data generators so we can apply transformations to only
        # the training data.
        train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, validation_split=0.20)
        valid_datagen = ImageDataGenerator(validation_split=0.20)

        train_iter = train_datagen.flow(data, labels, batch_size=cls._batch_size, shuffle=True, seed=123, subset="training")
        valid_iter = valid_datagen.flow(data, labels, batch_size=cls._batch_size, shuffle=True, seed=123, subset="validation")

        return train_iter, valid_iter
    
    @classmethod
    def get_test_set(cls, labels_to_keep):
        """Get a filtered test set, only containing data with the given labels."""
        data, labels = cls.filter_dataset(cls.X_test, cls.y_test, labels_to_keep)
        return data, labels

#CIFAR10.extract_class_data([0,1,2])
#CIFAR10.prepare_data(10)