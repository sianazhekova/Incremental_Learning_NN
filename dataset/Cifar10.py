import tensorflow as tf
from tensorflow.keras import datasets, layers

class CIFAR10:
    all_class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

    dataset_name = "Cifar 10"
    
    height = 32
    width = 32
    
    batch_size = 32 #args.batch_size 
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomCrop(32, 32),  #no option for padding in tf
        layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
        #tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    ])

    valid_data_augmentation = tf.keras.Sequential([
        #tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
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
    def prepare_cifar10_data_augment(data_augment, batch_size):
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
        X_train, y_train, X_test, y_test = X_train.batch(batch_size), y_train.batch(batch_size), X_test.batch(batch_size), y_test.batch(batch_size)
        X_train, X_valid = X_train[:-5000], X_train[-5000:]
        y_train, y_valid = y_train[:-5000], y_train[-5000:]
        
        train_ds = zip(X_train, y_train)
        valid_ds = zip(X_valid, y_valid)
        test_ds = zip(X_test, y_test)
        """
        train_ds = train_ds.batch(batch_size)
        valid_ds = valid_ds.batch(batch_size)
        test_ds = test_ds.batch(batch_size)
        """

        if data_augment:
            train_ds = train_ds.map(lambda x, y : (train_data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
            valid_ds = valid_ds.map(lambda x, y : (valid_data_augmentation(x), y), num_parallel_calls=AUTOTUNE)

        return train_ds.prefetch(buffer_size=AUTOTUNE), valid_ds.prefetch(buffer_size=AUTOTUNE), test_ds.prefetch(buffer_size=AUTOTUNE)
    
    @classmethod
    def import_data(cls,data_augment=False):
        train_ds, valid_ds, test_ds = cls.prepare_cifar10_data_augment(data_augment, cls.batch_size)
        X_train, y_train = ([x for x, y in train_ds], [y for x, y in train_ds])
        X_valid, y_valid = ([x for x, y in valid_ds], [y for x, y in valid_ds])
        X_test, y_test = ([x for x, y in test_ds], [y for x, y in test_ds])

        return X_train, y_train, X_valid, y_valid, X_test, y_test
        
