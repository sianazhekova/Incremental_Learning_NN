import sys
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import losses, layers, models, optimizers

from datasets.Cifar10 import CIFAR10

from diagnostics.IncrementalComparator import IncrementalComparator
from models.naive_cnn import plot_accuracy_loss_epoch

from . import naive_cnn as NaiveCNN
from models.module_nn import ModuleNN, OptimizerInputError


class iCaRL(ModuleNN):
    
    def iCarl_classify(self, x, state="Image"):
        """ Image or Feature Representation Classification of Model via Nearest-Mean-of-Exemplars Rule """
        # Input: x -> image to classify
        # Require: P := {P1, P2, ..., P|P|} set of exemplar sets
        # Require: feature_map -> underlying NN
        feature_map = self.icarl_model.layer[0]
        argmin_val = sys.maxint
        y_star = None
        
        model_prediction = None
        if state.lower() in ["image"]:
            model_prediction = tf.math.l2_normalize(feature_map.predict(x))
        if state.lower() == "feature":
            model_prediction = tf.math.l2_normalize(x)
        
        for key in self.P.keys():
            P_y =  self.P[key]
            mu = 1/len(P_y) * tf.math.reduce_sum([tf.math.l2_normalize(feature_map.predict(p)) for p in P_y], axis=0)  # mean of exemplars
            mu = tf.math.normalize(mu)

            abs_diff = abs(model_prediction - mu)
            if abs_diff < argmin_val:   # find nearest prototype
                argmin_val = abs_diff
                y_star = key
        
        return y_star
    

    def iCarl_loss_closure(self, alpha=1.00, beta=1.00):
        """ A Functional Closure for the Model Training Loss via Knowledge Distillation & Prototype Rehearsal """
        
        def iCarl_loss_fn(target_labels, model_outputs):  # y_actual, y_pred
            scalar_classes = tf.math.argmax(target_labels, axis=1)
            
            # TensorFlow does not have a nice substitute for numpy's isin().
            indices = tf.reduce_any(tf.equal(tf.expand_dims(scalar_classes, 1),
                                             self.new_labels), axis=1)

            # Filter based on whether each model output/class label is from the newly selected class labels
            print(~indices, target_labels)
            old_class_labels = tf.boolean_mask(target_labels, ~indices)
            new_class_labels = tf.boolean_mask(target_labels, indices)
            pred_old_classes = tf.boolean_mask(model_outputs, ~indices)
            pred_new_classes = tf.boolean_mask(model_outputs, indices)

            # Compute losses using these 4 variables as the (labels, logits) arguments to the 2 loss functions
            distillation_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=old_class_labels, logits=pred_old_classes)
            classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=new_class_labels, logits=pred_new_classes)

            return tf.reduce_mean(tf.concat([alpha * distillation_loss, beta * classification_loss], axis=0))
        
        return iCarl_loss_fn
    

    def iCarl_incremental_train(self, all_labels, new_labels, num_epochs, plot_verbose, loss_option=True):
        """ Design of the Main Stages Comprising the Incremental Training Strategy of iCaRL """
        # Input: X := {Xn, ..., Xt}  per-class sets of training samples of new classes
        # Require: K -> memory limit
        # Require: delta_param -> current model parameters
        # Require: P := {P1, ..., Pt} exemplar sets created so far
        
        old_labels = set(all_labels) - set(new_labels) 

        self.update_representation(new_labels, num_epochs, plot_verbose, loss_option)
        t = len(new_labels)
        m = self.K/t
        for label in old_labels:
            self.reduce_exemplar_set(label, m)
        
        # Create the X super-set of per-class sets of images
        X_new_set = OrderedDict()
        for new_label in new_labels:
            x_data, _  = self.dataset.custom_dataset_filter(new_label)
            X_new_set[new_label] = x_data

        for new_label in new_labels:
            self.construct_exemplar_set(X_new_set, new_label, m)
    

    def fit_GPU(self, num_epochs=200, plot_verbose=True):
        """ Run the Incremental Training Algorithm of iCaRL"""
        #if len(self.all_labels) == len(self.new_labels):
        all_labels = self.all_labels
        new_labels = self.new_labels
        self.iCarl_incremental_train(all_labels, new_labels, num_epochs, plot_verbose)


    def update_iterators_test_set(self, ds_class_name, labels, new_labels):
        """ Update the Member References to All Labels and the Newly Extracted Labels"""
        self.all_labels = labels
        self.new_labels = new_labels

        self.update_test_set(*ds_class_name.get_test_set(labels))
    

    def configure_optimizers(self, select_optimizer, lr=None, momentum=None, weight_decay=None):
        """ Selection of Optimizer Based on Indicated String Argument """
        if select_optimizer == "adam":
            lr = 0.01
            select_optimizer = optimizers.Adam(learning_rate=lr)
        elif select_optimizer == "SGDW":
            """ Hyper-parameter Setting for this option: lr = 0.1 , momentum = 0.9, weight_decay = 10^(-4) , epoch_num = 200, batch size = 256 """
            lr = 0.01
            num_steps = 80 * int(self.X_train.shape[0]/32)
            select_optimizer = tfa.optimizers.SGDW(learning_rate=optimizers.schedules.PiecewiseConstantDecay(boundaries=[num_steps, num_steps], values=[lr, (lr*0.1), (lr*0.1)]), momentum=0.9, weight_decay=1) #TODO: for SGDW, loss_fun should be sparse categorical cross-entropy(?)
        elif select_optimizer == "SGD":
            lr = 0.01
            num_steps = 80 * int(self.X_train.shape[0]/32)
            select_optimizer = optimizers.SGD(learning_rate=optimizers.schedules.PiecewiseConstantDecay(boundaries=[num_steps, num_steps], values=[lr, (lr*0.1), (lr*0.1)]), momentum=0.9)
        else:
            raise OptimizerInputError(select_optimizer, "Invalid argument string name of optimizer")
        
        return select_optimizer


    def custom_compile(self, loss='categorical_crossentropy', select_optimizer='adam', metrics_options=['accuracy']):
        """ Model Compilation Stage """
        # Selection of optimizer based on indicated string argument
        optimizer_to_use = self.configure_optimizers(select_optimizer)

        self.icarl_model.compile(optimizer=optimizer_to_use, loss=loss, metrics=metrics_options)
    

    def training_step(self, num_epochs, plot_verbose=True, loss_option=True, custom_bs=None):
        """ Model Training Stage """
        self.history = self.icarl_model.fit(self.train_iter, batch_size=custom_bs, epochs = num_epochs, validation_data=self.valid_iter, callbacks=[self.history])
        self.test_score = self.icarl_model.evaluate(self.X_test, self.y_test, verbose=2)
        print(self.history.history.keys())
        if plot_verbose:
            plot_accuracy_loss_epoch(self.history, self.icarl_model, num_epochs, loss_option)


    def update_iterators(self, train_iter, valid_iter):
        self.train_iter = train_iter
        self.valid_iter = valid_iter
    

    def update_representation(self, new_labels, num_epochs, plot_verbose, loss_option):
        """ Updating the Model Representation """
        # DONE: In construct_exemplar_set, only deal with the raw data - extracted images from the selected/additional labels
        # DONE: In update_representation, create training_iter, val_iter for the images from the newly added labels, BUT only create trainig_iter for the images from the exemplar classes
        # DONE: Zip them together and construct the new dataset D
        # DONE: Train the model, where the loss function arguments and parameters are specified as vectors 
        
        new_train_data, new_train_labels = self.dataset.filter_dataset(
        self.dataset.X_train, self.dataset.y_train, new_labels)

        _, h, w, c = new_train_data.shape

        # transform the P exemplar sets into an iterator (the exemplar iterator)
        all_exemplar_results = np.empty(shape=(0, self.dataset.get_default_num_classes()))
        all_exemplar_data = np.empty(shape=(0, h, w, c))
        for exemplar_label in self.P.keys():
            exemplar_X_set = self.P[exemplar_label]
            exemplar_results = self.icarl_model.predict(exemplar_X_set)
            all_exemplar_results = tf.concat([all_exemplar_results, exemplar_results], axis=0)
            all_exemplar_data = tf.concat([all_exemplar_data, exemplar_X_set], axis=0)
        VALID_SPLT = 0.20 # (!!!) to be included as an argument for the argument parser
        P_train_iter, P_valid_iter =  self.dataset.create_custom_iterators(all_exemplar_data, all_exemplar_results, valid_split=VALID_SPLT)
        
        all_train_data = np.concatenate((all_exemplar_data, new_train_data))
        all_train_labels = np.concatenate((all_exemplar_results, new_train_labels))

        # Create the combined dataset D via combining the iterators for the dataset entries for the new labels with the old ones 
        train_iter, valid_iter = self.dataset.create_custom_iterators(
            all_train_data, all_train_labels, valid_split=VALID_SPLT)

        self.update_iterators(train_iter, valid_iter)

        ### Run the model & update parameters (encompassed within the delta parameter) ###
        custom_bs = None  # 128

        if self.opt_GPU:
            with tf.device('/device:GPU:0'):
                self.training_step(num_epochs, plot_verbose, loss_option, custom_bs)
        else:
            self.training_step(num_epochs, plot_verbose, loss_option, custom_bs)
    

    def construct_exemplar_set(self, X_set, label, m):
        """ Construction of an Exemplar Set for a Specified Label """
        # Require: current feature function of base NN: feature_map 
        # Input: Raw Image Set X = {x1, x2, ...., xn} of class label
        # Input: m target number of exemplars for new class
        mu = 0
        n = len(X_set[label])
        feature_map_table = {}
        feature_map = self.icarl_model.layers[0]

        #enums_x_data = {} -> An alternative that can be used in case enumerate() does not iterate through the X set in the same order
        for enum_i, x in enumerate(X_set[label]):
            #enums_x_data[enum_i] = x
            feature_map_table[enum_i] = tf.math.l2_normalize(feature_map.predict(tf.expand_dims(x, axis=0)))[0]
            mu += feature_map_table[enum_i]
        
        mu = mu/n
        mu = tf.math.l2_normalize(mu)

        self.P[label] = np.array([])
        P_list = self.P[label]

        for k in range(1, m+1):
            argmin_val = sys.maxint
            pk = None
            exemplar_features_sum = tf.math.reduce_sum([tf.math.l2_normalize(feature_map.predict(tf.expand_dims(p, axis=0))[0]) for p in P_list[:k]], axis=0) # Check this !!!
            for enum_i, x in enumerate(X_set[label]):
                abs_diff = abs(mu - tf.math.l2_normalize(1/k * (feature_map_table[enum_i] + exemplar_features_sum)))   # CHECK THIS !!!
                if abs_diff < argmin_val:
                    argmin_val = abs_diff
                    pk = x
            P_list = np.append(P_list, [pk], axis=0)

        assert len(self.P[label]) == m
    

    def reduce_exemplar_set(self, y, m):
        """ Reduction of Exemplar Sets """
        # Input: m -> target number of exemplars
        # Py = {p1, p2, ..., p|Py|} current exemplar set
        P_new = np.array([])
        for p in self.P[y]:   # Here, we are selecting only the first m exemplars. We may implement another strategy for exemplar selection 
            P_new = np.append(P_new, [p], axis=0)
            m-=1
            if m == 0:
                break
        self.P[y] = P_new
    

    def __init__(self, GPU, ds_class_name, cls_model, K=2000):
        """ Model Constructor & Hyper-/Parameter Initialisation """
        super(iCaRL, self).__init__(GPU, ds_class_name)
        
        self.train_iter = None
        self.valid_iter = None

        self.P = {}
        self.default_num_labels = ds_class_name.get_default_num_classes()
        self.all_labels = None
        self.new_labels = None
        
        # Memory Limit = total number of exemplars
        self.K = K

        # Construct the iCaRL model by appending the feature map and the dense representation layer
        self.icarl_model = models.Sequential([cls_model.model.layers[0],
                                              layers.Dense(
                                                  units=self.default_num_labels,
                                                  activation=None,
                                                  name="incremental_classification_layer")]
                                            )
        # Compile the iCaRL model
        self.custom_compile(loss=self.iCarl_loss_closure())
    
if __name__ == "__main__":
    # This is the commands that I was executing leading to that error message
    naiveCNN = NaiveCNN.NaiveCNN(GPU=True, ds_class_name=CIFAR10)
    iCarlCNN = iCaRL(GPU=True, ds_class_name=CIFAR10, cls_model=naiveCNN)
    IncrementalComparator.evaluate_class_acc_score(iCarlCNN, CIFAR10, start_size=2, increment_size=2)
