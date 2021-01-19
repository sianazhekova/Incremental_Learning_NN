import os,sys
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)
sys.path.append("/content/gdrive/My Drive/IncrementalNN")

from tensorflow.keras import datasets, layers, models, losses, optimizers
import matplotlib.pyplot as plt
from functools import partial

sys.path.insert(1, os.path.join(sys.path[0], '..'))
print(tf.__version__)

import dataset
from dataset import Cifar10
from tensorflow.keras.callbacks import History 


# Global variables field
ConvNN2D = partial(layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

MaxPool2DPartial = partial(layers.MaxPooling2D, pool_size=2)
AvrPool2DPartial = partial(layers.AveragePooling2D, pool_size=(2, 2), strides=None, padding="valid", data_format=None)

def plot_accuracy_loss_epoch(history, model, num_epochs, loss_option=True):
    train_loss_score = history.history['loss']
    validation_loss_score = history.history['val_loss']
    
    train_acc_score = history.history['categorical_accuracy']
    validation_acc_score = history.history['val_categorical_accuracy']

    x_axis_epochs = range(num_epochs)
    plt.figure(figsize=(9,9))
    plt.subplot(1,2,1)
    plt.plot(x_axis_epochs, train_acc_score, label='Training Accuracy')
    plt.plot(x_axis_epochs, validation_acc_score, label='Validation Accuracy')
    plt.legend(loc='upper left')
    plt.title('Training and Validation Accuracy vs Number of Epochs')

    plt.subplot(1,2,2)
    plt.plot(x_axis_epochs, train_loss_score, label='Training Loss')
    plt.plot(x_axis_epochs, validation_loss_score, label='Validation Loss')
    plt.legend(loc='lower left')
    plt.title('Training and Validation Loss vs Number of Epochs')
    plt.show()


class NaiveCNN:

    def construct_cnn_v2(self):
        model = models.Sequential([
            ConvNN2D(filters=64, kernel_size=3, input_shape=[self.img_height, self.img_width, self.num_channels]),
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
            layers.Flatten(),
            layers.Dense(units=10, activation='softmax')
        ])
        model.summary()
        return model

    def compile_fit_model(self, loss_fun, select_optimizer, metrics_options, num_epochs, plot_verbose=True, loss_option=True):
        #try with "adam" optimiser as well, metrics = ["sparse_categorical_accuracy"]
        self.model.compile(optimizer=select_optimizer, loss=loss_fun, metrics=metrics_options)
        print(f"The length of the training dataset numpyarray iterator is {self.train_iter.__len__()}")
        self.history = self.model.fit(self.train_iter, epochs = num_epochs, validation_data=(self.X_valid, self.y_valid), callbacks=[self.history])
        self.test_score = self.model.evaluate(self.X_test, self.y_test, verbose=2)  #test_loss,test_acc -> will need these for the sequential class addition performance evaluation
        print(self.history.history.keys())
        if plot_verbose:
            plot_accuracy_loss_epoch(self.history, self.model, num_epochs, loss_option)

    def compile_fit_GPU(self, plot_verbose=True):
        loss_fun = self.args.loss_fn 
        select_optimizer = self.args.optimizer
        metrics_options = [self.args.metrics]
        num_epochs = self.args.num_epochs
        if select_optimizer == "adam":
          lr = args.learning_rate #0.01
          select_optimizer = optimizers.Adam(learning_rate=lr)
        if select_optimizer == "SGDW":  # lr = 0.1 , momentum = 0.9, weight_decay = 10^(-4) , epoch_num = 200
            lr = args.learning_rate
            #len_ds = len(X_train)+len(X_valid)+len(X_test)+len(y_train)+len(y_valid)+len(y_test)
            #num_steps = 80*(len_ds/args.batch_size)
            num_steps = 80 * int(self.X_train.shape[0]/args.batch_size)
            select_optimizer = tfa.optimizers.SGDW(learning_rate=optimizers.schedules.PiecewiseConstantDecay(boundaries=[num_steps, num_steps], values=[lr, (lr*0.1), (lr*0.1)]), momentum=args.momentum, weight_decay=args.weight_decay) #TODO: for SGDW, loss_fun should be sparse categorical cross-entropy(?)
        if select_optimizer == "SGD":
            lr = args.learning_rate
            num_steps = 80 * int(self.X_train.shape[0]/args.batch_size)
            select_optimizer = optimizers.SGD(learning_rate=optimizers.schedules.PiecewiseConstantDecay(boundaries=[num_steps, num_steps], values=[lr, (lr*0.1), (lr*0.1)]), momentum=args.momentum)
        if self.opt_GPU:
            with tf.device('/device:GPU:0'):
                self.compile_fit_model(loss_fun, select_optimizer, metrics_options, num_epochs, plot_verbose)
        else:
            self.compile_fit_model(loss_fun, select_optimizer, metrics_options, num_epochs, plot_verbose)

    def __init__(self, GPU, args, ds_class_name):
        
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = ds_class_name.import_data()
        self.dataset = ds_class_name.dataset_name
        self.num_channels = ds_class_name.get_num_channels()
        self.img_height = ds_class_name.height
        self.img_width = ds_class_name.width

        self.data_generator, self.train_iter = ds_class_name.get_data_generator(self.X_train, self.y_train, self.X_valid, self.y_valid)

        self.model = self.construct_cnn_v2()
        self.opt_GPU = GPU
        self.args = args
        self.history = History()
        if self.opt_GPU:
            device_name = tf.test.gpu_device_name()
            if device_name != '/device:GPU:0':
                raise SystemError('GPU device not found')
            print('Found GPU at: {}'.format(device_name))


def args_parse():
    parser = argparse.ArgumentParser(description='Supply naive CNN with hyper-parameter configuration & options')
    # will add for architecture/model used for the multiple models used
    #The parameters for our model compilation
    compile_env = parser.add_argument_group(title="Model Compilation")
    compile_env.add_argument('--optimizer', '-o', default='adam', type=str, help='Choice of an optimzer to use when compiling the model during training and validation')
    compile_env.add_argument('--learning-rate', '-lr', default=0.01, type=float, help='Select a learning rate for the model to use when updating weights during training')
    compile_env.add_argument('--loss-fn', '-lf', default='categorical_crossentropy', type=str, help='Choise of a loss function to minimise and use during update step')
    compile_env.add_argument('--metrics', '-me', default='categorical_accuracy', type=str, help='Metric to use for model compilation')
    compile_env.add_argument('--momentum', '-mo', default=0.9, type=float, help='Momentum value to use in the special case of a Stochastic Gradient Descent with weight decay')
    compile_env.add_argument('--weight_decay', '-wd', default=1, type=float, help='Weight decay value to use in the special case of a Stocastic Gradient Descent with weight decay')

    #The parameters for our training dataset
    train_ds_env = parser.add_argument_group(title="Parameters for Training Dataset")
    train_ds_env.add_argument('--batch-size', '-b', default=32, type=int, help='Number of data instances per batch when multi-batching')
    train_ds_env.add_argument('--num-epochs', '-e', default=200, type=int, help='Number of epochs to use during training')
    train_ds_env.add_argument('--data-augment', '-da', default=False, type=bool, help='Option for including data augmentation for the training and validation datasets.')

    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = args_parse()

    naiveCNN = NaiveCNN(GPU=False, args=args, ds_class_name=dataset.Cifar10.CIFAR10)
    naiveCNN.compile_fit_GPU()
    test_loss, test_acc = naiveCNN.test_score
    print(f"The Loss for our model & test dataset is {test_loss} and the Accuracy for our model & test dataset is {test_acc} ")

    