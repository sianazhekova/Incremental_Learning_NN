import tensorflow as tf
from tensorflow.keras import datasets, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CIFAR10:
    all_class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

    dataset_name = "Cifar 10"
    
    height = 32
    width = 32
    
    batch_size = 256 #args.batch_size 

    @staticmethod
    def get_default_num_classes():
        return len(all_class_names)
    
    @staticmethod
    def get_num_channels():
        return 3

    @staticmethod
    def scale_pixels(train, valid, test):
        train_norm = train.astype('float32')
        valid_norm = valid.astype('float32')
        test_norm = test.astype('float32')
        
        train_norm, valid_norm, test_norm = train_norm/255.0, valid_norm/255.0, test_norm/255.0

        return train_norm, valid_norm, test_norm
    
    @staticmethod
    def prepare_data():
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
        X_train, X_valid = X_train[:-5000], X_train[-5000:]
        
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        y_train, y_valid = y_train[:-5000], y_train[-5000:]

        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    @classmethod
    def import_data(cls):
        X_train, X_valid, X_test, y_train, y_valid, y_test = cls.prepare_data()
        X_train, X_valid, X_test = cls.scale_pixels(X_train, X_valid, X_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    @classmethod
    def get_data_generator(cls, X_train, y_train, X_valid, y_valid):
        generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        train_iter = generator.flow(X_train, y_train, batch_size=cls.batch_size)

        return generator, train_iter

        
