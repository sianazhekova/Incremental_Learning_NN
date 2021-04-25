""" Abstract Class (protocol to follow between dataset subclasses) """

from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical


class Dataset(ABC):
    _all_class_names = None

    _dataset_name = None

    _height = None
    _width = None
    _num_channels = None

    _batch_size = None

    @classmethod
    @abstractmethod
    def get_default_num_classes(cls):
        return len(cls._all_class_names)
    
    @classmethod
    @abstractmethod
    def get_img_height(cls):
        return cls._height
    
    @classmethod
    @abstractmethod
    def get_img_width(cls):
        return cls._width
    
    @classmethod
    @abstractmethod
    def get_num_channels(cls):
        return cls._num_channels

    @classmethod
    @abstractmethod
    def get_batch_size(cls):
        return cls._batch_size
    
    @classmethod
    @abstractmethod
    def get_dataset_name(cls):
        return cls._dataset_name
    
    @classmethod
    def combine_generators(cls, gen, other_gen):
        while True:
            yield next(gen), next(other_gen)
    
    @classmethod
    def create_custom_iterators(cls, data, labels, valid_split):
        print(f"The labels are {labels} and their dims are {labels.size}")
        if data.size == 0:
            data = tf.constant([], shape=(0, 0, 0, 0))
        if labels.size != 0:  # the list of labels is not empty
            scalar_classes = tf.math.argmax(labels, axis=1)
            print(
                f"The scalar classes are {scalar_classes} and their dims are {scalar_classes.shape}")
            labels = to_categorical(scalar_classes,
                                    num_classes=cls.get_default_num_classes())
        else:
            labels = tf.constant([], shape=(0, 0, 0, 0))
        # print(f"The newly structured labels are {labels} and their dims are {labels.size}")
        train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           horizontal_flip=True,
                                           validation_split=valid_split)
        valid_datagen = ImageDataGenerator(validation_split=valid_split)

        train_iter = train_datagen.flow(data, labels,
                                        batch_size=cls._batch_size,
                                        shuffle=True, seed=123,
                                        subset="training")
        valid_iter = valid_datagen.flow(data, labels,
                                        batch_size=cls._batch_size,
                                        shuffle=True, seed=123,
                                        subset="validation")

        return train_iter, valid_iter

    
    @staticmethod
    @abstractmethod
    def scale_pixels(dataset):
        pass
    
    @classmethod
    @abstractmethod
    def filter_dataset(cls, data, labels, labels_to_keep):
        pass
    
    @classmethod
    @abstractmethod
    def custom_dataset_filter(cls, labels_to_keep):
        pass