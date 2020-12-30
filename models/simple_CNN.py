import argparse
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
from functools import partial

# Global variables field
ConvNN2D = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

MaxPool2DPartial = partial(layers.MaxPooling2D, pool_size=2)
AvrPool2DPartial = partial(layers.AveragePooling2D, pool_size=(2, 2), strides=None, padding="valid", data_format=None)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

class NaiveCNN:
    def initialise_datasets(self, import_ds):
        X_train, y_train, self.X_test, self.y_test = import_ds()
        self.X_train, self.X_valid = X_train[:-5000], X_train[-5000:]
        self.y_train, self.y_valid = y_train[:-5000], y_train[-5000:]

    def construct_cnn_v1(*act_args_units=['relu', 'relu', 'softmax', 128, 64, 10]):
        n = len(act_args_units)//2
        activation_args = act_args_units[:n]
        units = act_args_units[n:]
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
        model.summary() #-> to check output dimensionality
        return model

    def construct_cnn_v2():
        model = models.Sequential([
            ConvNN2D(filters=64, kernel_size=3, input_shape=[32, 32, 3]),
            ConvNN2D(filters=64, kernel_size=3),
            ConvNN2D(filters=128, kernel_size=3, strides=2),
            ConvNN2D(filters=128, kernel_size=3),
            layers.Dropout(0.5),
            ConvNN2D(filters=128, kernel_size=3),
            ConvNN2D(filters=192, kernel_size=3, strides=2),
            ConvNN2D(filters=192, kernel_size=3),
            layers.Dropout(0.5),
            ConvNN2D(filters=192, kernel_size=3),
            AvrPool2DPartial(kernel_size=8),
            ConvNN2D(filters=class_num, kernel_size=1, padding="valid")
        ])
        model.summary()
        return model

    def compile_fit_model(self, loss_fun, select_optimizer, metrics_options, num_epochs, plot_verbose, loss_option):
        self.model.compile(loss=loss_fun, optimizer=select_optimizer, metrics=metrics_options)   #try with "adam" optimiser as well, metrics = ["sparse_categorical_accuracy"]
        self.model.compile(optimizer=select_optimizer, loss=loss_fun, metrics=metrics_options)
        self.history = self.model.fit(self.X_train, self.y_train, epochs = num_epochs, validation_data=(self.X_valid, self.y_valid))
        self.test_score = self.model.evaluate(self.X_test, self.y_test, verbose=2)  #test_loss,test_acc -> will need these for the sequential class addition performance evaluation
        #y_pred = self.model.predict(self.X_test)
        if plot_verbose:
            plot_accuracy_loss_epoch(self.history, self.model, num_epochs, loss_option)


    def compile_fit_GPU(self, opt_GPU, loss_fun=args.loss_fun, select_optimizer=args.optimizer, metrics_options=["accuracy"], num_epochs=args.num_epochs, plot_verbose=True):
        if select_optimizer == "SGDW":
            select_optimizer = tfa.optimizers.SGDW(learning_rate=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) #TODO: for SGDW, loss_fun should be sparse categorical cross-entropy(?)
        if opt_GPU:
            with tf.device('/device:GPU:0'):
                compile_fit_model(loss_fun, select_optimizer, metrics_options, plot_verbose)
        else:
            compile_fit_model(loss_fun, select_optimizer, metrics_options, plot_verbose)

    def __init__(self, which_ver = "Version 2", construct_cnn = construct_cnn_v1, opt_GPU=True):
        if which_ver == "Version 1":
            self.model = construct_cnn(loss_fun, select_optimizer, metrics_options)
        else:
            self.model = construct_cnn_v2()


def args_parse(dataset):
    parser = argparse.ArgumentParser(description='Supply naive CNN with hyper-parameter configuration & options')
    
    

    train_env = parser.add_argument_group(title="Training parameters")
    train_env.add_argument()

    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supply naive CNN with hyper-parameter configuration & options for the ')
    parser.add_argument('-', '--', ) 

    