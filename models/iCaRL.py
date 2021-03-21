# TO-DO: create an abstract class for NaiveCNN and iCaRL that we can wrap functionality requisite for the incremental comparator class with iCarl
import math

class iCaRL(ModuleNN):
    def __init__(self, GPU, dataset, model, K=2000):
        super(iCaRL, self).__init__(GPU, dataset)
        self.K = k  # Memory Limit = total number of exemplars
        self.feature_map = model.layers[0]
        self.classification_layer = layers.Dense(units=len(self.num_exemplar_classes), activation='sigmoid', label="incremental_classification_layer") #model.layers[-1]
        self.delta_weights = None
        
        self.P = {}
        #self.P_train_iter = None
        #self.P_valid_iter = None
        #self.temp_train_iter = None
        #self.temp_test_iter = None
    
    def icarl_loss_fn(self, y_true, y_pred, ):
        # y_true = [batch_size, d0, .. dN],  y_pred = [batch_size, d0, .. dN]
        """
        for (x_batch, y_batch) in comb_iter:
            q = {}
            for exemplar_batch in exemplar_iter:
            total_loss = 0
            # Classification Step
            for new_y_batch in temp_iter:
                g = self.comb_model.predict(x_batch)  # y_pred
                total_loss += (y_batch == new_y_batch)*math.log(g) + (y_batch != new_y_batch)*math.log(1- g)
            # Distillation Step
            for exemplar_y_batch in exemplar_iter:
                g = """
        
        total_loss = 0

        # Classification Step
        for new_y_batch in temp_iter:
            total_loss += (y_true == new_y_batch)*math.log(g) + (y_batch != new_y_batch)*math.log(1- g)
        # Distillation Step
        for exemplar_y_batch in exemplar_iter:
            total_loss += 
            



        return loss_fn
    
    def cons_delta_map(self, model):
        # Construct the delta parameter by appending the base NN with a dense layer of wieghts with sigmoid activations
        sequential_model = Sequential().add(model)
        num_exemplar_classes = self.P.values()
        sequential_model.add(layers.Dense(units=len(num_exemplar_classes), activation='sigmoid', label="incremental_classification_layer"))
        sequential_model.compile(optimiser=, loss=self.icarl_loss_fn)

        return sequential_model

    def iCarl_classify_image(self, x):
        # Input: x -> image to classify
        # Require: P := {P1, P2, ..., P|P|} set of exemplar sets
        # Require: feature_map -> underlying NN
        argmin_val = sys.maxint
        y_star = None
        
        for key in self.P.keys():
            P_y =  self.P[key]
            mu = 1/len(P_y) * sum([feature_map.predict(p) for p in P_y])  # mean of exemplars
            
            abs_diff = abs(feature_map.predict(x) - mu)
            if abs_diff < argmin_val:   # find nearest prototype
                argmin_val = abs_diff
                y_star = key
        
        return y_star

    
    def iCarl_incremental_train(self, X_train, delta_param, all_labels, new_labels):
        # Input: X := {Xn, ..., Xt}  per-class sets of training samples of new classes
        # Require: K -> memory limit
        # Require: delta_param -> current model parameters
        # Require: P := {P1, ..., Pt} exemplar sets created so far
        
        old_labels = set(all_labels) - set(new_labels) 

        self.update_representation(X_train, old_labels, new_labels)
        t = len(new_labels)
        m = self.K/t
        for label in old_labels:
            self.reduce_exemplar_set(label, m)
        
        for new_label in new_labels:
            self.construct_exemplar_set(X_train, new_label, m)
    
    def update_iterators(self, train_iter, valid_iter):
        self.temp_train_iter, self.temp_valid_iter = train_iter, valid_iter
    
    def compile_fit_GPU(self, num_epochs, plot_verbose):

    """
    @staticmethod
        def increment_class_set(ds_class_name, model, labels, new_labels, loss_arr, acc_arr):
            model.update_iterators_test_set(ds_class_name, labels, new_labels)
                model.update_iterators(*ds_class_name.get_iterators(new_labels))
                model.update_test_set(*ds_class_name.get_test_set(labels))

            model.compile_fit_GPU(num_epochs=50, plot_verbose=False)
            base_test_loss, base_test_acc = model.get_test_loss_acc()
            loss_arr.append(base_test_loss)
            acc_arr.append(base_test_acc) """
    
    def update_iterators_test_set(self, ds_class_name, labels, new_labels):
        combined_data_iters = combine_generators()
        self.update_test_set(*ds_class_name.get_test_set(labels))


    def update_representation(self, X_train, old_labels, new_labels): # this function is going inside compile_fit_model()
        # TO-DO: In construct_exemplar_set, only deal with the raw data - extracted images from the selected/additional labels 
        # TO-DO: In update_representation, create training_iter, val_iter for the images from the newly added labels, BUT only create trainig_iter for the images from the exemplar classes
        # TO-DO: Zip them together and construct the new dataset D
        # TO-DO: Train the model, where the loss function arguments and parameters are specified as vectors 
        
        
        # transform the P exemplar sets into an iterator (the exemplar iterator)
        # combine the data associated with the new labels with the data from the exemplar sets (D)
        # turn that into an iterator (shuffle it for the train_iter and allocate for valid_iter)

        # the data associated with the new data extracted with them -> transform them into iterators
        

        # Create the combined dataset D via combining the iterators for the dataset entries for the new labels with the old ones 
        #comb_train_iter = self.combine_generators(old_train_iter, new_train_iter)
        #comb_valid_iter = self.combine_generators(old_valid_iter, new_valid_iter)

        
        




    
    def construct_exemplar_set(self, X_set, label, m):
        # Require: current feature function of base NN: feature_map 
        # Input: Raw Image Set X = {x1, x2, ...., xn} of class label
        # Input: m target number of exemplars for new class
        mu = 0
        n = len(X_set[label])
        feature_map_table = {}

        for x in in X_set[label]:
            feature_map_table[x] = self.feature_map.predict(x)
            mu += feature_map_table[x]
        
        mu = mu/n
        self.P[label] = {}
        P_set = self.P[label]
        for k in range(1, m+1):
            argmin_val = sys.maxint
            pk = None
            for x in X_set[label]:
                sum _feature_maps
                abs_diff = abs(mu - 1/k * feature_map_table[x] + sum([feature_map.predict(p) for p in P_set]))
                if abs_diff < argmin_val:
                    argmin_val = abs_diff
                    pk = x
            P_set.add(pk)
        

        assert self.P[label] == m
    
    def reduce_exemplar_set(self, y, m):
        # Input: m -> target number of exemplars
        # Py = {p1, p2, ..., p|Py|} current exemplar set
        P_new = {}
        for p in self.P[y]:   # Here, we are selecting only the first m exemplars. We may implement another strategy for exemplar selection 
            P_new.add(p)
            m-=1
            if m == 0:
                break
        self.P[y] = P_new

