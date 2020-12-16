import argparse
import tensorflow as tf

from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
from functools import partial

# Global variables field
ConvNN2D = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

MaxPool2DPartial = partial(layers.MaxPooling2D, pool_size=2)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def import_cifar10_dataset():
    X_train, y_train, X_test, y_test = datasets.cifar10.load_data()
    X_train, X_test = X_train/255.0, X_test/255.0

    return X_train, y_train, X_test, y_test

c10_X_train, c10_y_train, c10_X_test, c10_y_test = import_cifar10_dataset()
c10_X_train, c10_X_valid = c10_X_train[:-5000], c10_X_train[-5000:]
c10_y_train, c10_y_valid = c10_y_train[:-5000], c10_y_train[-5000:]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#copied the plot_figure() code from tensorflow website to visualise image classes, will probably delete it later
def plot_figure(num_images=25, fig_size=10):
    plt.figure(figsize=(fig_size,fig_size))
    total_size = (c10_X_train.__sizeof__ + c10_y_train.__sizeof__)
    if num_images > total_size:
        num_size = min(total_size, 25)
    for i in range(num_images):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

def plot_accuracy_epoch(history, model):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend*(loc='lower right')

class NaiveCNN:
    def initialise_datasets(self, import_dataset_fun, ):
    
    def compile_fit_model(self, loss_fun="sparse_categorical_crossentropy", select_optimizer="nadam", metrics_options=["accuracy"]):
        self.model.compile(loss=loss_fun, optimizer=select_optimizer, metrics=metrics_options)   #try with "adam" optimiser as well, metrics = ["sparse_categorical_accuracy"]
        """model.compile(optimizer='adam', SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])"""
        self.history = self.model.fit(self.X_train, self.y_train, epochs = 10, validation_data=(self.X_valid, self.y_valid))
        self.test_score = self.model.evaluate(self.X_test, self.y_test, verbose=2)  #test_loss,test_acc
        #y_pred = self.model.predict(self.X_test)

    def construct_cnn_v1(self, opt_GPU=True, *activation_args=('relu', 'relu', 'softmax'), *units=(128, 64, 10), loss_fun="sparse_categorical_crossentropy", select_optimizer="nadam", metrics_options=["accuracy"]):
        n = len(activation_args)
        assert(n == len(units))
        model = models.Sequential([
            ConvNN2D(filters=64, kernel_size=7, input_shape=[32, 32, 3]), #format of cifar images = (32, 32, 3)
            layers.MaxPooling2D(pool_size=2),
            ConvNN2D(filters=128),
            ConvNN2D(filters=128),
            layers.MaxPooling2D(pool_size=2),
            ConvNN2D(filters=256),
            ConvNN2D(filters=256),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten()])  #Flatten (unroll) the 2D output to 1D
            for i in range(n):
                model.add(layers.Dense(units= units[i], activation=activation_args[i]))
                if i != (n-1):
                    model.add(layers.Dropout(0.5))
        ])
        #model.summary() -> to check output dimensionality

        if opt_GPU:
            with tf.device('/device:GPU:0'):
                compile_fit_model()
        else:
            compile_fit_model()

    def __init__(self, opt_GPU=True, *activation_args=('relu', 'relu', 'softmax'), *units=(128, 64, 10), loss_fun="sparse_categorical_crossentropy", select_optimizer="nadam", metrics_options=["accuracy"]):
        self.model = 




""" TODO: 1. Implement a slightly different ordering of the layers for the CNN construction 
          2. Check if it works
          3. Play with different hyper-parameters
          4. Start implementing a subroutine, plotting accuracy vs number of classes added(!!!)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    