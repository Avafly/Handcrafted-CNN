import sys
import numpy as np

from model import nn_classifier
from keras.datasets import mnist

#########################
## dataset preparation ##
#########################

print("Loading MNIST ...", flush=True)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# insert channel dimension of 1
# X: (n, 28, 28) -> (n, 1, 28, 28)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# divide training and validation data
# 50,000 training and 10,000 validation samples
n_train_sample = 50000

# preprocessing
# normalize pixel values to (0, 1)
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

###########################
## hyperparameters setup ##
###########################

lr = 1.5e-3
n_epoch = 1
batch_size = 64

# decay (beta) for RMSProp
beta = 0.9

# define classifier
# classifier = nn_mnist_classifier(rmsprop_beta=beta, lr=lr)
classifier = nn_classifier("./model_config.json", rmsprop_beta=beta, lr=lr)

# number of steps per epoch
n_steps = int(n_train_sample / batch_size)

# split data into training and validation dataset
X_split = np.split(X_train, [n_train_sample, ])
X_trn = X_split[0]
X_val = X_split[1]

y_split = np.split(y_train, [n_train_sample, ])
y_trn = y_split[0]
y_val = y_split[1]

# set this to True if we want to compute validation scores
do_validation = True

# set this to True if we want to plot sample predictions
plot_sample_prediction = True

# show the info of data and hyperparameters
print("=" * 45)
print("Training data shape:", X_trn.shape)
print("Training labels shape:", y_trn.shape)
print("Validation data shape:", X_val.shape)
print("Validation labels shape:", y_val.shape)
print("Test data shape:", X_test.shape)
print("Test label shape:", y_test.shape)
print("Learning rate:", lr)
print("Number of epochs:", n_epoch)
print("Batch size:", batch_size)
print("=" * 45)

##############
## training ##
##############

for i in range(n_epoch):
    # randomly shuffle training data
    shuffled_index_train = np.arange(y_trn.shape[0])
    np.random.shuffle(shuffled_index_train)

    # shuffled for randomized mini-batch
    X_trn = X_trn[shuffled_index_train]
    y_trn = y_trn[shuffled_index_train]

    print("Epoch number: %d/%d" % (i + 1, n_epoch), " "*1)

    # for tracking training accuracy
    trn_accy = 0

    for j in range(n_steps):
        # take mini-batch from training set
        X = X_trn[j * batch_size : (j+1) * batch_size, ]
        y = y_trn[j * batch_size : (j+1) * batch_size, ]

        # perform forward, backprop and weight update
        scores, loss = classifier.forward(X, y)
        classifier.backprop()
        classifier.update_weights()

        # for tracking training accuracy
        estim = np.ravel(np.argmax(scores, axis=1))
        trn_accy += np.sum((estim==y).astype("uint8")) / batch_size

        # check loss every 50 loops
        if (j + 1) % 50 == 0:
            print("Loss: %.4f" % loss, " "*5)
            
            # print training accuracy every 200 loops
            if (j + 1) % 200 == 0:
                print("Training accuracy: %.2f%%" % (trn_accy / 2))
                trn_accy = 0

                # evaluate the validation accuracy
                if do_validation:
                    # pick 100 random samples frm validation dataset
                    val_idx = np.random.randint(low=0, high=y_val.shape[0], size=(100,))

                    X = X_val[val_idx]
                    y = y_val[val_idx]

                    # take random batch of batch_size
                    scores, loss = classifier.forward(X, y, is_training=False)
                    estim = np.ravel(np.argmax(scores, axis=1))

                    # compare softmax vs y
                    val_accy = np.sum((estim==y).astype("uint8"))
                    print("Validation accuracy: %.2f%%" % val_accy)

        # show progress
        if (j + 1) % 10 == 0:
            print("Progress: [%5.1f%%]" % ((j+n_steps*i) / n_steps / n_epoch * 100), end="\r")
            sys.stdout.flush()

#############
## testing ##
#############
# test_batch: accuracy is measured in this batch size
# test_iter: total number of batch iterations to complete testing over test data
# tot_accy: total accuracy

print("Start testing", " "*4)

test_batch = 1000
test_iter = int(y_test.shape[0] / test_batch)
tot_accy = 0

for i in range(test_iter):
    X = X_test[i * test_batch : (i+1) * test_batch, ]
    y = y_test[i * test_batch : (i+1) * test_batch, ]

    # forward pass
    scores, loss = classifier.forward(X, y, is_training=False)
    estim = np.ravel(np.argmax(scores, axis=1))
    accy = np.sum((estim==y).astype("uint8")) / test_batch
    tot_accy += accy
    print("Batch accuracy: %.2f%%" % (accy * 100))

# print out final accuracy
print("Total accuracy: %.5f%%" % (tot_accy / test_iter * 100))

# test plot randomly picked 10 samples
if plot_sample_prediction:
    import matplotlib.pyplot as plt
    
    num_plot = 10
    sample_index = np.random.randint(0, X_test.shape[0], (num_plot,))
    plt.figure(figsize=(12, 4))

    for i in range(num_plot):
        idx = sample_index[i]
        img = np.squeeze(X_test[idx])
        ax = plt.subplot(1, num_plot, i + 1)
        plt.imshow(img, cmap=plt.get_cmap("gray"))

        X = X_test[idx : idx+1]
        y = y_test[idx : idx+1]

        # get prediction from clissifier
        score, _ = classifier.forward(X, y, is_training=False)
        pred = np.ravel(np.argmax(score, axis=1))

        # if prediction is correct, the title will be in black
        # o/w. the title will be in red
        if y_test[idx] == pred:
            title_color = "k"
        else:
            title_color = "r"

        ax.set_title("Label:" + str(y_test[idx]) + "\nPred: " + str(pred[0]), color=title_color)

    plt.tight_layout()
    plt.savefig("test_result.png")
    plt.show()