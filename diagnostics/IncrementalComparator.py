import matplotlib as plt
import random
import tensorflow as tf

tf.keras.backend.clear_session

class IncrementalComparator:
    def __init__(self, lo, hi):
        self.map = dict()
        self.lower_limit, self.upper_limit = lo, hi
        self.M = hi
    
    def next(self):
        if (self.M == self.lower_limit -1):
            self.map = dict()
            self.M = self.upper_limit
        n = random.randint(self.lower_limit, self.M)
        ret_val = self.map.get(n, n)
        self.map[n] = self.map.get(self.M, self.M)
        self.M -= 1
        return ret_val
    
    @classmethod
    def unit_test(cls, lo, hi, range_n=1):
        random.seed(0)
        model = cls(lo, hi)
        for i in range(hi - lo + range_n):
            print(model.next())
    
    @staticmethod
    def increment_class_set(ds_class_name, model, labels, new_labels, loss_arr, 
                            acc_arr):
        model.update_iterators_test_set(ds_class_name, labels, new_labels)

        model.fit_GPU(num_epochs=50, plot_verbose=False)
        base_test_loss, base_test_acc = model.get_test_loss_acc()
        loss_arr.append(base_test_loss)
        acc_arr.append(base_test_acc)
    
    @staticmethod
    def plot_acc_loss_class(class_acc, acc_arr, loss_acc=None):
        plt.figure(figsize=(9, 9))
        plt.plot(class_acc, acc_arr, '-bo', label='Testing Accuracy')
        
        plt.legend(loc='lower right')
        plt.title('Testing Accuracy vs Number of Classes')
        plt.show()
        
    @classmethod
    def evaluate_class_acc_score(cls, model, ds_class, start_size=2,
                                increment_size=2):
        random.seed(0)
        # generator = model.get_data_generator()
        print("HERE")
        num_classes = ds_class.get_default_num_classes()
        print(f"Default numnber of classes is {num_classes}")
        lo, hi = 0, (num_classes - 1)

        label_generator = cls(lo, hi)
        labels = []
        for i in range(start_size):
            new_class = label_generator.next()
            labels.append(new_class)
        print(labels)
        model.update_iterators_test_set(ds_class, labels, labels)

        model.fit_GPU(num_epochs=200, plot_verbose=False)
        base_test_loss, base_test_acc = model.get_test_loss_acc()
        loss_arr = [base_test_loss]
        acc_arr = [base_test_acc]
        class_arr = [start_size]

        diff = (num_classes - start_size)
        steps =  diff // increment_size
        for i in range(steps):
            new_labels = []
            for j in range(increment_size):
                new_class = label_generator.next()
                new_labels.append(new_class)
                labels.append(new_class)
            print(new_labels)
            cls.increment_class_set(ds_class, model, labels, new_labels,
                                    loss_arr, acc_arr)
            class_arr.append(start_size + increment_size * (i + 1))
        
        remaining = diff - steps * increment_size
        if (remaining > 0):
            labels = []
            for i in range(remaining):
                new_class = label_generator.next()
                new_labels.append(new_class)
                labels.append(new_class)
            cls.increment_class_set(ds_class, model, labels, new_labels,
                                    loss_arr, acc_arr)
            class_arr.append(num_classes)
        
        cls.plot_acc_loss_class(class_arr, acc_arr, loss_arr)
