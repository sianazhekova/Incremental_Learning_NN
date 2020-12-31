

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

