import os
import sys
from functools import partial

import matplotlib.pyplot as plt

import argparse
from utils.argument_parser import args_parse
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers, models, losses, optimizers
from models.module_nn import ModuleNN

sys.path.insert(1, os.path.join(sys.path[0], '..'))
print(tf.__version__)

import datasets


# Global variables field
ConvNN2D = partial(layers.Conv2D, kernel_size=3, activation='relu', 
                    padding="SAME")

MaxPool2DPartial = partial(layers.MaxPooling2D, pool_size=2)
AvrPool2DPartial = partial(layers.AveragePooling2D, pool_size=8, strides=None, 
                            padding="valid", data_format=None)

def plot_accuracy_loss_epoch(history, model, num_epochs, loss_option=True):
    train_loss_score = history.history['loss']
    validation_loss_score = history.history['val_loss']
    
    train_acc_score = history.history['accuracy']
    validation_acc_score = history.history['val_accuracy']

    x_axis_epochs = range(num_epochs)
    plt.figure(figsize=(9, 9))
    plt.subplot(1, 2, 1)
    plt.plot(x_axis_epochs, train_acc_score, label='Training Accuracy')
    plt.plot(x_axis_epochs, validation_acc_score, label='Validation Accuracy')
    plt.legend(loc='upper left')
    plt.title('Training and Validation Accuracy vs Number of Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(x_axis_epochs, train_loss_score, label='Training Loss')
    plt.plot(x_axis_epochs, validation_loss_score, label='Validation Loss')
    plt.legend(loc='lower left')
    plt.title('Training and Validation Loss vs Number of Epochs')
    plt.show()


class NaiveCNN(ModuleNN):

    def construct_cnn_v2(self):
        """ Construction & Layering of CNN"""
        feature_extractor = models.Sequential([
            ConvNN2D(filters=64, kernel_size=3,
                    input_shape=[self.img_height, self.img_width, 
                                self.num_channels]),
            layers.BatchNormalization(),
            ConvNN2D(filters=64, kernel_size=3),
            layers.BatchNormalization(),
            ConvNN2D(filters=128, kernel_size=3, strides=2),
            layers.BatchNormalization(),
            ConvNN2D(filters=128, kernel_size=3),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            ConvNN2D(filters=128, kernel_size=3),
            layers.BatchNormalization(),
            ConvNN2D(filters=192, kernel_size=3, strides=2),
            layers.BatchNormalization(),
            ConvNN2D(filters=192, kernel_size=3),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            ConvNN2D(filters=192, kernel_size=3),
            layers.BatchNormalization(),
            AvrPool2DPartial(),
            layers.Flatten()
        ], name="feature_map")
        classification_layer = layers.Dense(units=self.default_num_labels, 
                                            activation='softmax',
                                            name="simple_classification_layer")
        
        model = models.Sequential([feature_extractor, classification_layer])
        model.summary()
        return model

    def configure_optimizers(self, select_optimizer, lr=None, momentum=None,
                            weight_decay=None):
        """ Selection of Optimizer Based on Indicated String Argument """
        if select_optimizer == "adam":
            lr = 0.01
            select_optimizer = optimizers.Adam(learning_rate=lr)
        elif select_optimizer == "SGDW":
            """ Hyper-parameter Setting for this option: lr = 0.1 , momentum = 0.9, weight_decay = 10^(-4) , epoch_num = 200, batch size = 256 """
            lr = 0.01
            num_steps = 80 * int(self.X_train.shape[0] / 32)
            select_optimizer = tfa.optimizers.SGDW(
                learning_rate=optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[num_steps, num_steps],
                    values=[lr, (lr*0.1), (lr*0.1)]), momentum=0.9,
                weight_decay=1) #TODO: for SGDW, loss_fun should be sparse categorical cross-entropy(?)
        elif select_optimizer == "SGD":
            lr = 0.01
            num_steps = 80 * int(self.X_train.shape[0] / 32)
            select_optimizer = optimizers.SGD(
                learning_rate=optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[num_steps, num_steps],
                    values=[lr, (lr*0.1), (lr*0.1)]), momentum=0.9)
        else:
            raise OptimizerInputError(select_optimizer,
                                    "Invalid argument string name of optimizer")
        
        return select_optimizer

    def custom_compile(self, loss='categorical_crossentropy', 
                        select_optimizer='adam', metrics_options=['accuracy']):
        """ Model Compilation Stage """
        # Selection of optimizer based on indicated string argument
        optimizer_to_use = self.configure_optimizers(select_optimizer)

        
        self.model.compile(optimizer=optimizer_to_use, loss=loss,
                        metrics=metrics_options)
    

    def training_step(self, num_epochs, plot_verbose=True, loss_option=True,
                    custom_bs=None):
        """ Model Training Stage """
        print(
            f"The length of the training dataset numpyarray iterator is {self.train_iter.__len__()}")
        if custom_bs == None:
            custom_bs = self.dataset.get_batch_size()
        self.history = self.model.fit(self.train_iter, batch_size=custom_bs,
                                    epochs = num_epochs,
                                    validation_data=self.valid_iter,
                                    callbacks=[self.history])
        self.test_score = self.model.evaluate(self.X_test, self.y_test,
                                            verbose=2)  #test_loss,test_acc -> will need these for the sequential class addition performance evaluation
        print(self.history.history.keys())
        if plot_verbose:
            plot_accuracy_loss_epoch(self.history, self.model, num_epochs, 
                                    loss_option)


    def fit_GPU(self, num_epochs=200, plot_verbose=True):
        custom_bs = self.dataset.get_batch_size()

        if self.opt_GPU:
            with tf.device('/device:GPU:0'):
                self.training_step(num_epochs, plot_verbose, custom_bs)
        else:
            self.training_step(num_epochs, plot_verbose, custom_bs)
    

    def update_iterators_test_set(self, ds_class_name, labels, new_labels):
        self.update_iterators(*ds_class_name.get_iterators(new_labels))
        self.update_test_set(*ds_class_name.get_test_set(labels))


    def update_iterators(self, train_iter, valid_iter):
        self.train_iter = train_iter
        self.valid_iter = valid_iter
    

    def __init__(self, args, ds_class_name):
        self.train_iter = None
        self.valid_iter = None

        super(NaiveCNN, self).__init__(GPU, ds_class_name)

        if ds_class_name != None:
            self.default_num_labels = self.dataset.get_default_num_classes()
        
        self.model = self.construct_cnn_v2()
        self.custom_compile()

        if self.opt_GPU:
            device_name = tf.test.gpu_device_name()
            if device_name != '/device:GPU:0':
                raise SystemError('GPU device not found')
            print('Found GPU at: {}'.format(device_name))

tf.keras.backend.clear_session

if __name__ == "__main__":
    naiveCNN = NaiveCNN(GPU=True, ds_class_name=CIFAR10)
    naiveCNN.generate_iterators()
    naiveCNN.compile_fit_GPU()

    test_loss, test_acc = naiveCNN.test_score
    print(f"The Loss for our model & test dataset is {test_loss} and the Accuracy for our model & test dataset is {test_acc} ")