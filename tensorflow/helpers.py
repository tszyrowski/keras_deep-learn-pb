# Helper functions for reuse
import matplotlib.pyplot as plt
import numpy as np

print(plt.style.available)

def plot_loss_acc_simple(history):
    """Plots loss and accuracy."""
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, "bo", label="Training loss") 
    plt.plot(epochs, val_loss_values, "r", label="Validation loss") 
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()    
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "r", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def plot_loss_acc(history):
    """Plots loss and accuracy."""
    history_dict = history.history

    loss_values = history_dict["loss"]
    if "val_loss" in history_dict.keys():
        loss = "val_loss"
        loss_label = "Validation loss"
    elif "loss" in history_dict.keys():
        loss = "loss"
        loss_label = "Loss"
    val_loss_values = history_dict[loss]

    acc = history_dict["accuracy"]
    if "val_accuracy" in history_dict.keys():
        accuracy = "val_accuracy"
        accuracy_label = "Validation Accuracy"
    elif "accuracy" in history_dict.keys():
        accuracy = "accuracy"
        accuracy_label = "Accuracy"
    val_acc = history_dict[accuracy]


    epochs = range(1, len(loss_values) + 1)
    with plt.style.context('seaborn-v0_8'):
            
        # Create a figure and a grid of subplots
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        
        # Plotting training and validation loss on the first subplot
        ax[0].plot(epochs, loss_values, "bo-", label="Training loss")
        ax[0].plot(epochs, val_loss_values, "r-", label=loss_label)
        ax[0].set_title(f"Training and {loss_label}")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[0].grid(True)
        ax[0].minorticks_on()
        ax[0].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

        # Plotting training and validation accuracy on the second subplot
        ax[1].plot(epochs, acc, "bo-", label="Training Accuracy")
        ax[1].plot(epochs, val_acc, "r-", label=accuracy_label)
        ax[1].set_title(f"Training and {accuracy_label}")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        ax[1].grid(True)
        ax[1].minorticks_on()
        ax[1].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

        # Setting the title for the entire figure
        plt.suptitle("Training and Validation Metrics", fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # padding around subplots.
        plt.show()

def vectorize_sequences(sequences, dimension=10000):
    """Vectorize to one-hot encoding
    
    Usage:
        samp_num = 1
        x_train = vectorize_sequences(train_data)
        x_test = vectorize_sequences(test_data)

        print(
            f"Create arrays of {x_train.shape} as and example:\n",
            f"x_train length of {len(x_train[samp_num])} ",
            "and first 100 datapoints:\n",
            f"{x_train[samp_num][:100]}\n",
            f"it has  {np.count_nonzero(x_train[samp_num])} of ones"
        )
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

def decode_to_words(sample, word_index):
    """Decode indices back to words
    
    Note: The word index required to be recovered before:
    >>> word_index = imdb.get_word_index()
    Or
    >>> word_index = reuters.get_word_index()
    """
    reverse_word_idx = dict([(val, key) for (key, val) in word_index.items()])
    decoded_review = " ".join([reverse_word_idx.get(i-3,"?") for i in sample])
    return decoded_review