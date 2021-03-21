
from abc import ABC, abstractmethod

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
    def combine_generators(gen, other_gen):
        while True:
            yield(next(gen), next(other_gen))
    
    @classmethod
    @abstractmethod
    def create_custom_generator():

    @staticmethod
    @abstractmethod
    def scale_pixels(dataset):
        pass
    
    @classmethod
    @abstractmethod
    def filter_dataset(cls, data, labels, labels_to_keep):
        pass
