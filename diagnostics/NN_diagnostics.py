

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

def plot_accuracy_loss_epoch(history, model, num_epochs, loss_option=True):
    train_loss_score = history.history['loss']
    validation_loss_score = history.history['val_loss']
    
    train_acc_score = history.history['accuracy']
    validation_acc_score = history.history['val_accuracy']

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