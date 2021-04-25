# Abstract Superclass for NiaveCNN, iCaRL, EWC and any other models extending these defining the protocol of model initialisation & loading 
from abc import ABC, abstractmethod

from tensorflow.python.keras.callbacks import History


class ModuleNN(ABC):
    
    @abstractmethod
    def __init__(self, GPU, ds_class_name):
        self.test_score = None
        
        self.X_train = None 
        self.y_train = None 
        self.X_test = None 
        self.y_test = None

        self.dataset = ds_class_name
        self.num_channels = ds_class_name.get_num_channels()
        self.img_height = ds_class_name.get_img_height()
        self.img_width = ds_class_name.get_img_width()

        self.default_num_labels = None

        self.opt_GPU = GPU
        self.history = History()
    
    @abstractmethod
    def configure_optimizers(self, select_optimizer, lr, momentum, 
                            weight_decay):
        pass

    @abstractmethod
    def training_step(self, num_epochs, plot_verbose, loss_option, custom_bs):
        pass

    @abstractmethod
    def fit_GPU(self, num_epochs, plot_verbose):
        pass
    
    def get_test_loss_acc(self):
        return self.test_score

    def update_datasets(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
    
    def generate_iterators(self, labels_to_keep=None):
        if labels_to_keep == None:
            num_classes = self.dataset.get_default_num_classes()
            labels_to_keep = range(num_classes)
        train_iter, valid_iter = self.dataset.get_iterators(labels_to_keep)
        self.train_iter, self.valid_iter = train_iter, valid_iter

    @abstractmethod
    def update_iterators(self, train_iter, valid_iter):
        pass
    
    def update_test_set(self, test_data, test_labels):
        self.X_test = test_data
        self.y_test = test_labels

# This will be transferred into the training folder
   
class Error(Exception):
    """ Base class for exception in this project """
    pass

class OptimizerInputError(Error):
    """ Exception raised for errors in the input specifying the optimizer to use
    
    Attributes:
       opt_str --- input optimizer string in which the error occurred
       message --- explanation of the error
    """

    def __init__(self, opt_str, message):
        self.opt_str = opt_str
        self.message = message