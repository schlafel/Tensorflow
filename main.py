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

    print(30*"*")
    # define cnn model
    def define_model():
        # We begin by defining the a empty stack. We'll use this for building our
        # network, later by layer.
        model = tf.keras.models.Sequential()

        # We start with a convolutional layer this will extract features from
        # the input images by sliding a convolution filter over the input image,
        # resulting in a feature map.
        model.add(
            tf.keras.layers.Conv2D(
                filters=32,  # How many filters we will learn
                kernel_size=(3, 3),  # Size of feature map that will slide over image
                strides=(1, 1),  # How the feature map "steps" across the image
                padding='valid',  # We are not using padding
                activation='relu',  # Rectified Linear Unit Activation Function
                input_shape=(28, 28, 1)  # The expected input shape for this layer
            )
        )

        # The next layer we will add is a Maxpooling layer. This will reduce the
        # dimensionality of each feature, which reduces the number of parameters that
        # the model needs to learn, which shortens training time.
        model.add(
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),  # Size feature will be mapped to
                strides=(2, 2)  # How the pool "steps" across the feature
            )
        )

        # We'll now add a dropout layer. This fights overfitting and forces the model to
        # learn multiple representations of the same data by randomly disabling neurons
        # in the learning phase.
        model.add(
            tf.keras.layers.Dropout(
                rate=0.25  # Randomly disable 25% of neurons
            )
        )

        # Output from previous layer is a 3D tensor. This must be flattened to a 1D
        # vector before beiung fed to the Dense Layers.
        model.add(
            tf.keras.layers.Flatten()
        )

        # A dense (interconnected) layer is added for mapping the derived features
        # to the required class.
        model.add(
            tf.keras.layers.Dense(
                units=128,  # Output shape
                activation='relu'  # Rectified Linear Unit Activation Function
            )
        )

        # Final layer with 10 outputs and a softmax activation. Softmax activation
        # enables me to calculate the output based on the probabilities.
        # Each class is assigned a probability and the class with the maximum
        # probability is the model’s output for the input.
        model.add(
            tf.keras.layers.Dense(
                units=10,  # Output shape
                activation='softmax'  # Softmax Activation Function
            )
        )

        # Build the model
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,  # loss function
            optimizer=tf.keras.optimizers.Adam(),  # optimizer function
            metrics=['accuracy']  # reporting metric
        )

        # Display a summary of the models structure
        model.summary()
        return model


    model_RCN = define_model()

    history = model_RCN.fit(training_images, training_labels, epochs=10,
                            # batch_size=32,
                            validation_data=(test_images, test_labels))
    # evaluate model
    _, acc = model_RCN.evaluate(test_images, test_labels, verbose=0)
    print('> %.3f' % (acc * 100.0))


    plot_history(history)




    plt.show()











