import numpy as np
# import nn_layers module make by me
import nn_layers as nnl

class nn_mnist_classifier:
    def __init__(self, rmsprop_beta=0.9, lr=1.0e-2):
        # for saving intermediate variables used for backprop
        self.fwd_cache = None

        ## initialize each layer

        # convolutional layer 1
        self.conv_layer_1 = nnl.nn_convolutional_layer(kernel_size=3, in_ch_size=1, out_ch_size=32)

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
        # epsilon
        self.epsilon = 1e-5

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
        epsilon = self.epsilon

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