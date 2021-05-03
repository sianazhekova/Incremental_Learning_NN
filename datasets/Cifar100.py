from tensorflow.keras import datasets
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from datasets.Dataset import Dataset

flatten = lambda x : [item for sublist in x for item in sublist] 

class CIFAR100(Dataset):
    
    _hierarchical_dict = {
        'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
        'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
        'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
        'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
        'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
        'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
        'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
        'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
        'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
        'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
        'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
        'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
        'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
        'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
        'people': ['baby', 'boy', 'girl', 'man', 'woman'],
        'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
        'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
        'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
        'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    }

    _all_class_names = flatten (_hierarchical_dict.values())

    _dataset_name = "Cifar 10"
    
    _height = 32
    _width = 32
    _num_channels = 3
    
    _batch_size = 256 #args.batch_size
    
    (X_train, y_train), (X_test, y_test) = datasets.cifar100.load_data()
    
    @staticmethod
    def scale_pixels(dataset):
        return dataset.astype('float32')/255.0
    
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

        train_iter = train_datagen.flow(data, labels, batch_size=cls._batch_size, shuffle=True, seed=123, subset="training")
        valid_iter = valid_datagen.flow(data, labels, batch_size=cls._batch_size, shuffle=True, seed=123, subset="validation")

        return train_iter, valid_iter
    
    @classmethod
    def get_test_set(cls, labels_to_keep):
        """Get a filtered test set, only containing data with the given labels."""
        data, labels = cls.filter_dataset(cls.X_test, cls.y_test, labels_to_keep)
        return data, labels
