import tensorflow as tf
from tensorflow.keras import datasets, layers

class CIFAR10:
    all_class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

    dataset_name = "Cifar 10"
    
    batch_size = 32
    AUTOTUNE = tf.data.experiemntal.AUTOTUNE

    train_data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomCrop(32, 32),  #no option for padding in tf
        layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
        layers.experimental.preprocessing.Normalization(mean=[0.4914, 0.4822,0.4465], variance=[0.2023, 0.1994, 0.2010])
    ])

    val_data_augmentation = tf.keras.Sequemtial([
        layers.experimental.preprocessing.Normalization(mean=[0.4914, 0.4822,0.4465], variance=[0.2023, 0.1994, 0.2010])
    ])

    @staticmethod
    def get_default_num_classes():
        return len(all_class_names)
    
    @staticmethod
    def get_num_channels():
        return 3

    @staticmethod
    def import_cifar10_with_scaling():
        X_train, y_train, X_test, y_test = datasets.cifar10.load_data()
        X_train, X_valid = X_train[:-5000], X_train[-5000:]
        y_train, y_valid = y_train[:-5000], y_train[-5000:]
        
        X_train, X_test = X_train/255.0, X_test/255.0

        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    @staticmethod
    def import_cifar10_data_augment():
        X_train, y_train, X_test, y_test = datasets.cifar10.load_data()
        X_train, X_valid = X_train[:-5000], X_train[-5000:]
        y_train, y_valid = y_train[:-5000], y_train[-5000:]

        X_train = X_train.map(lambda x : train_data_augmentation(x), num_parallel_calss=AUTOTUNE)
        
        

"""
@staticmethod
    def train_loader(num_workers, batch_size, distributed):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cifar10._normalize,
        ])

        _normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                      std=[0.2023, 0.1994, 0.2010])

"""