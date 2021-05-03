# Implementation of MNIST dataset

from tensorflow.keras import datasets
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from datasets.Dataset import Dataset


class MNIST(Dataset):
    _all_class_names = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    _dataset_name = "MNIST"

    _height = 28
    _width = 28
    _num_channels = 1

    _batch_size = 256

    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    @staticmethod
    def scale_pixels(dataset):
        return dataset.astype('float32')/255.0
    
    @classmethod
    def filter_dataset(cls, data, labels, labels_to_keep):
        """Filter a dataset to contain only data with the specified labels."""
        data = cls.scale_pixels(data)
        #print(f"The dataset has a shape {data.shape}")
        # Filter data.
        keep_indices = np.isin(labels, labels_to_keep)
        filtered_data = data[keep_indices]
        print(f"Filtered data shape is: {filtered_data.shape}")
        filtered_labels = labels[keep_indices]
        filtered_labels = to_categorical(filtered_labels, num_classes=cls.get_default_num_classes())

        return filtered_data, filtered_labels
    
    @classmethod
    def custom_dataset_filter(cls, labels_to_keep):
        """ Filter from the underlying dataset images and labels according using the specified labels argument  """
        data, labels = cls.filter_dataset(cls.X_train, cls.y_train, labels_to_keep)
        return data, labels

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

        data = tf.expand_dims(data, axis=1)
        train_iter = train_datagen.flow(data, labels, batch_size=cls._batch_size, shuffle=True, seed=123, subset="training")
        valid_iter = valid_datagen.flow(data, labels, batch_size=cls._batch_size, shuffle=True, seed=123, subset="validation")

        return train_iter, valid_iter
    
    @classmethod
    def get_test_set(cls, labels_to_keep):
        """Get a filtered test set, only containing data with the given labels."""
        data, labels = cls.filter_dataset(cls.X_test, cls.y_test, labels_to_keep)
        return data, labels