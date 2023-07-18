import numpy as np

# import nn_layers module make by me
import nn_layers as nnl

class nn_mnist_classifier:
    def __init__(self, rmsprop_beta=0.9, lr=1.0e-2):
        # for saving intermediate variables used for backprop
        self.fwd_cache = None

        ## initialize each layer

        # convolutional layer 1
        self.conv_layer_1 = nnl.nn_convolutional_layer(Wx_size=3, Wy_size=3, in_ch_size=1, out_ch_size=32)

        # activation layer
        self.act_1 = nnl.nn_activation_layer_relu()

        # maxpool
        self.maxpool_layer_1 = nnl.nn_max_pooling_layer(stride=2, pool_size=2)

        # fully connected layer 1
        self.fc_layer_1 = nnl.nn_fc_layer(input_size=13*13*32, output_size=128)
        self.act_2 = nnl.nn_activation_layer_relu()

        # fully connected layer 2
        self.fc_layer_2 = nnl.nn_fc_layer(input_size=128, output_size=10)

        # softmax
        self.sm1 = nnl.nn_softmax_layer()

        # cross entropy
        self.xent = nnl.nn_cross_entropy_layer()

        # for RMSProp parameter initialization
        self.is_first_update = True

        # decay (beta) for RMSProp update
        self.rmsprop_beta = rmsprop_beta
        # learning rate
        self.lr = lr

    # forward method
    # inputs:
    #   x: input data in batch
    #   y: labels of the batch
    #   is_training: True if backprop method is called next
    #                False if only forward pass (inference) needed
    def forward(self, X, y, is_training=True):
        # cv1_f, ac1_f, mp1_f, fc1_f, ac2_f, fc2_f, sm1_f, cn_f
        # are outputs from each layer of the CNN
        # cv1_f is the output from the convolutional layer
        # ac1_f is the output from 1st activation layer
        # mp1_f is the output from maxpooling layer
        # ... and so on
        cv1_f = self.conv_layer_1.forward(X, is_training)

        ac1_f = self.act_1.forward(cv1_f, is_training)
        mp1_f = self.maxpool_layer_1.forward(ac1_f, is_training)

        fc1_f = self.fc_layer_1.forward(mp1_f, is_training)
        ac2_f = self.act_2.forward(fc1_f, is_training)

        fc2_f = self.fc_layer_2.forward(ac2_f, is_training)

        sm1_f = self.sm1.forward(fc2_f, is_training)
        cn_f = self.xent.forward(sm1_f, y, is_training)  # cn_f is the loss of the current input batch
        
        scores = sm1_f
        loss = cn_f

        # store required only when is_training is True
        if is_training:
            self.fwd_cache = (X, y)

        # forward will return scores (sm1_f) and loss (cn_f)
        return scores, loss

    def backprop(self):
        (X, y) = self.fwd_cache

        cn_b = self.xent.backprop(y)
        sm1_b = cn_b

        fc2_b, dldw_fc2, dldb_fc2 = self.fc_layer_2.backprop(sm1_b)
        ac2_b = self.act_2.backprop(fc2_b)

        fc1_b, dldw_fc1, dldb_fc1 = self.fc_layer_1.backprop(ac2_b)
        mp1_b = self.maxpool_layer_1.backprop(fc1_b)

        ac1_b = self.act_1.backprop(mp1_b)
        cv1_b, dldw_cv1, dldb_cv1 = self.conv_layer_1.backprop(ac1_b)

    def update_weights(self):

        beta = self.rmsprop_beta
        lr = self.lr
        epsilon = 1e-8

        # load dLdW and dLdb for weight update
        dldw_fc2, dldb_fc2 = self.fc_layer_2.get_gradients()
        dldw_fc1, dldb_fc1 = self.fc_layer_1.get_gradients()
        dldw_cv1, dldb_cv1 = self.conv_layer_1.get_gradients()

        ####################
        ## RMSProp update ##
        ####################

        # initialize v_w and v_b if it is first time update
        if self.is_first_update:
            self.v_w_fc2 = np.zeros_like(dldw_fc2)
            self.v_b_fc2 = np.zeros_like(dldb_fc2)

            self.v_w_fc1 = np.zeros_like(dldw_fc1)
            self.v_b_fc1 = np.zeros_like(dldb_fc1)

            self.v_w_cv1 = np.zeros_like(dldw_cv1)
            self.v_b_cv1 = np.zeros_like(dldb_cv1)

            self.is_first_update = False

        # calculate v for convolutional and FC layer updates
        self.v_w_fc2 = beta*self.v_w_fc2 + (1-beta)*np.square(dldw_fc2)
        self.v_b_fc2 = beta*self.v_b_fc2 + (1-beta)*np.square(dldb_fc2)

        self.v_w_fc1 = beta*self.v_w_fc1 + (1-beta)*np.square(dldw_fc1)
        self.v_b_fc1 = beta*self.v_b_fc1 + (1-beta)*np.square(dldb_fc1)

        self.v_w_cv1 = beta*self.v_w_cv1 + (1-beta)*np.square(dldw_cv1)
        self.v_b_cv1 = beta*self.v_b_cv1 + (1-beta)*np.square(dldb_cv1)

        # using v, perform weight update for each layer
        self.fc_layer_2.update_weights(dLdW=-lr*dldw_fc2/(np.sqrt(self.v_w_fc2)+epsilon),
                                       dLdb=-lr*dldb_fc2/(np.sqrt(self.v_b_fc2)+epsilon))

        self.fc_layer_1.update_weights(dLdW=-lr*dldw_fc1/(np.sqrt(self.v_w_fc1)+epsilon),
                                       dLdb=-lr*dldb_fc1/(np.sqrt(self.v_b_fc1)+epsilon))

        self.conv_layer_1.update_weights(dLdW=-lr*dldw_cv1/(np.sqrt(self.v_w_cv1)+epsilon),
                                         dLdb=-lr*dldb_cv1/(np.sqrt(self.v_b_cv1)+epsilon))


#########################
## dataset preparation ##
#########################

# load MNIST dataset
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# load fashion MNIST dataset
# from keras.datasets import fashion_mnist
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# insert channel dimension of 1
# X: (n, 28, 28) -> (n, 1, 28, 28)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# divide training and validation data
# 50,000 training and 10,000 validation samples
n_train_sample = 50000

# preprocessing
# normalize pixel values to (0, 1)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

###########################
## hyperparameters setup ##
###########################

lr = 1.5e-3
n_epoch = 1
batch_size = 64

# decay (beta) for RMSProp
beta = 0.9

# define classifier
classifier = nn_mnist_classifier(rmsprop_beta=beta, lr=lr)

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

    print("Epoch number: %d/%d" % (i + 1, n_epoch))

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
            print("Progress: %.2f%%" % ((j+n_steps*i) / n_steps / n_epoch * 100))
            print("Loss: %.4f" % loss)
            
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

            print("-" * 30)

#############
## testing ##
#############
# test_batch: accuracy is measured in this batch size
# test_iter: total number of batch iterations to complete testing over test data
# tot_accy: total accuracy

print("Start testing")

test_batch = 100
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

        ax.set_title("Label:" + str(y_test[idx]) + "\n Pred:" + str(int(pred)), color=title_color)

    plt.tight_layout()
    plt.savefig("test_result.png")
    plt.show()