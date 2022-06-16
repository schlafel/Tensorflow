import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np




class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.95): # Experiment with changing this value
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True


def plot_history(history):
    fig, ax = plt.subplots(1)
    ax.plot(history.epoch, history.history["val_loss"], label="Val-Loss")
    ax.plot(history.epoch, history.history["loss"], label="loss")
    ax2 = ax.twinx()

    ax2.plot(history.epoch, history.history["accuracy"], label="Accuracy", linestyle="--")
    ax2.plot(history.epoch, history.history["val_accuracy"], label="val_accuracy", linestyle="--")

    ax.legend()
    ax2.legend()
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy [-]")
    plt.tight_layout()
    ax.set_xticks(history.epoch)


    return fig,(ax,ax2)


def plot_it(train_images, train_labels, random_seed=42):
    label_dict = dict({0: "T-shirt/top",
                       1: "Trouser",
                       2: "Pullover",
                       3: "Dress",
                       4: "Coat",
                       5: "Sandal",
                       6: "Shirt",
                       7: "Sneaker",
                       8: "Bag",
                       9: "Ankle Boot"})

    idx_imgs = np.random.choice(range(0, len(train_images)), 9,
                                replace=False)

    fig, axs = plt.subplots(3, 3)
    axi = axs.flatten()
    for i in range(0, 9):
        print(idx_imgs[i])
        axi[i].imshow(train_images[idx_imgs[i]], cmap="gray")
        axi[i].set_title(label_dict[train_labels[idx_imgs[i]]])

        axi[i].axis("off")
    plt.tight_layout()

if __name__ == '__main__':
    print(tf.__version__)

    # Load the Fashion MNIST dataset
    fmnist = tf.keras.datasets.fashion_mnist



    # Load the training and test split of the Fashion MNIST dataset
    (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
    print("loaded data")

    # Normalize the pixel values of the train and test images
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    print("Shape of test_images:", test_images.shape)
    print("Shape of test_labels:", test_labels.shape)

    #val - dataset

    idx_val = np.random.choice(range(0,len(test_images)),int(len(test_images)*.3),
                               replace=False)

    val_images = test_images[idx_val]
    val_labels = test_labels[idx_val]

    #remove the idx_val from test-set
    test_images = np.delete(test_images,idx_val,axis = 0)
    test_labels = np.delete(test_labels,idx_val,axis = 0)


    #check the shapes
    print("Shape of val_images:", val_images.shape)
    print("Shape of val_labels:", val_labels.shape)
    print("Shape of test_images:", test_images.shape)
    print("Shape of test_labels:", test_labels.shape)




    # Build the classification model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate = .1),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer = tf.optimizers.Adam(),
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #initialize callback
    callbacks = myCallback()
    history = model.fit(training_images, training_labels,
                        validation_data=(test_images, test_labels),
                        epochs=5,
                        callbacks = [callbacks])

    # Evaluate the model on unseen data
    model.evaluate(test_images, test_labels)



    plot_history(history)



    print("Validation:", model.evaluate(val_images,val_labels))

    plt.show()











