import numpy as np
from skimage.util.shape import view_as_windows

#########################
## convolutional layer ##
#########################

class nn_convolutional_layer:
    def __init__(self, Wx_size, Wy_size, in_ch_size, out_ch_size, pad_size=0, std=1):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                 (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.pad_size = pad_size
        # fwd_cache is assigned values only when is_training==True in the forward method
        self.fwd_cache = None
        self.bwd_cache = None

    def update_weights(self, dLdW, dLdb):
        self.W += dLdW
        self.b += dLdb

    def get_weights(self):
        return self.W, self.b
    
    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def get_gradients(self):
        return self.bwd_cache["dLdW"], self.bwd_cache["dLdb"]

    def forward(self, X, is_training=True):
        # input shape
        batch_size, in_ch_size, inR, inC = X.shape
        # weight shape
        _, _, Wx_size, Wy_size = self.W.shape

        # determine the output dimensions
        outR = inR - Wx_size + 1 + 2 * self.pad_size
        outC = inC - Wy_size + 1 + 2 * self.pad_size

        # pad the input
        padding = [(0, 0), (0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size)]
        X_padded = np.pad(X, padding, mode='constant')

        # create x_windows for conv operation
        window_shape = (1, 1, Wx_size, Wy_size)
        view_window_stride = 1
        x_windows = view_as_windows(X_padded, window_shape, view_window_stride)
        x_windows = x_windows.reshape(batch_size, in_ch_size, outR, outC, Wx_size, Wy_size)

        # perform convolution operation
        # For each Wx_size*Wy_size window in 'x_windows' and each Wx_size*Wy_size
        # kernel in 'W', compute the element-wise multiplication and sum the
        # results. The operation is applied across all windows and kernels,
        # resulting in an output of shape (128, 28, 26, 26) which corresponds
        # to the convolution result for each sample, each output channel, and
        # each output location.
        out = np.tensordot(x_windows, self.W, axes=([1,4,5], [1,2,3]))
        # (batch_size, outR, outC, out_ch_size) -> (batch_size, out_ch_size, outR, outC)
        out = np.transpose(out, (0, 3, 1, 2))
        out += self.b

        # save the forward cache for backprop
        if is_training:
            # store intermediate variables
            self.fwd_cache = {}
            self.fwd_cache["X"] = X
            self.fwd_cache["X_padded"] = X_padded
            self.fwd_cache["out_shape"] = out.shape

        return out

    def backprop(self, dLdy):
        # ensure is_training=True, i.e., the forward cache exists
        assert self.fwd_cache != None

        # reshape dLdy to (batch_size, ch_size, outR, outC)
        # since the shape of dLdy may be (batch_size, ch_size * outR * outC)
        dLdy = dLdy.reshape(self.fwd_cache["out_shape"])

        # load the cache data
        X = self.fwd_cache["X"]
        X_padded = self.fwd_cache["X_padded"]
        pad_size = self.pad_size

        # input shape
        batch_size, in_ch_size, inR, inC = X.shape
        # weight shape
        out_ch_size, _, Wx_size, Wy_size = self.W.shape

        # determine the output dimensions
        outR = inR - Wx_size + 1 + 2 * pad_size
        outC = inC - Wy_size + 1 + 2 * pad_size

        # create x_windows for conv operation
        window_shape = (1, 1, Wx_size, Wy_size)
        view_window_stride = 1
        x_windows = view_as_windows(X_padded, window_shape, view_window_stride)
        x_windows = x_windows.reshape(batch_size, in_ch_size, outR, outC, Wx_size, Wy_size)

        # gradient w.r.t. weights
        dLdW = np.tensordot(dLdy, x_windows, axes=([0,2,3], [0,2,3]))

        # gradient w.r.t. bias
        dLdb = np.sum(dLdy, axis=(0,2,3)).reshape(1, out_ch_size, 1, 1)

        # gradient w.r.t. inputs
        # create dLdy_windows for conv operation
        # calculate unpad_size used in backprop
        # Since outR = inR-Wx_size+1+2*pad_size, inR = outR+Wx_size-1-2*pad_size => unpad_size_R = Wx_size-pad_size-1
        unpad_size_R = Wx_size - pad_size - 1
        unpad_size_C = Wy_size - pad_size - 1
        dLdy_padded = np.pad(dLdy, ((0, 0), (0, 0), (unpad_size_R, unpad_size_R), (unpad_size_C, unpad_size_C)), mode="constant", constant_values=0)
        dLdy_windows = view_as_windows(dLdy_padded, window_shape, view_window_stride)
        dLdy_windows = dLdy_windows.reshape(batch_size, out_ch_size, inR, inC, Wx_size, Wy_size)
        # create flipped W, i.e., rotate self.W 180 degrees
        W_flipped = self.W[..., ::-1, ::-1]

        # perform convolution operation, same as forward propagation
        dLdx = np.tensordot(dLdy_windows, W_flipped, axes=([1,4,5], [0,2,3]))
        dLdx = np.transpose(dLdx, (0, 3, 1, 2))

        # cache upstream gradients for weight updates
        self.bwd_cache = {}
        self.bwd_cache["dLdW"] = dLdW
        self.bwd_cache["dLdb"] = dLdb

        return dLdx, dLdW, dLdb


#######################
## max pooling layer ##
#######################

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        self.fwd_cache = None

    def forward(self, X, is_training=True):
        # input shape
        batch_size, in_ch_size, inR, inC = X.shape

        # get the output shape after pooling
        outR = (inR - self.pool_size) // self.stride + 1
        outC = (inC - self.pool_size) // self.stride + 1

        # create a window view of the input tensor and then max pool
        x_windows = view_as_windows(X, (1, 1, self.pool_size, self.pool_size), step=(1, 1, self.stride, self.stride))
        x_reshaped = x_windows.reshape(batch_size, in_ch_size, outR, outC, self.pool_size, self.pool_size)
        out = x_reshaped.max(axis=(4, 5))  # max pool

        # if is_training, save mask for grident calculation in backpropagation phase
        if is_training:
            out_expanded = out.repeat(self.pool_size, axis=2).repeat(self.pool_size, axis=3)
            x_window = X[:, :, :outR*self.stride, :outC*self.stride]
            mask = np.equal(x_window, out_expanded).astype(int)
            # store intermediate variables
            self.fwd_cache = {}
            self.fwd_cache["X"] = X
            self.fwd_cache["mask"] = mask
            self.fwd_cache["out_shape"] = out.shape
            # TODO
            # A window may have multiple maxima, causing the gradient to explode during backward propagation.
            # It is better to make sure that there is only one maximum index in a window.

        return out

    def backprop(self, dLdy):
        # ensure is_training=True, i.e., the forward cache exists
        assert self.fwd_cache != None

        # reshape dLdy to (batch_size, ch_size, outR, outC)
        # since the shape of dLdy may be (batch_size, ch_size * outR * outC)
        dLdy = dLdy.reshape(self.fwd_cache["out_shape"])

        dLdy_expanded = dLdy.repeat(self.pool_size, axis=2).repeat(self.pool_size, axis=3)
        # compute the gradient using mask obtained in forward propagation
        dLdx = np.multiply(self.fwd_cache["mask"], dLdy_expanded)
        return dLdx


###########################
## fully connected layer ##
###########################
# fully connected linear layer
# parameters: weight matrix W and bias b
# forward computation of y = Wx + b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size, )

class nn_fc_layer:
    def __init__(self, input_size, output_size, std=1):
        # Xavier/He initializer
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b = 0.01 + np.zeros((output_size))
        self.fwd_cache = None
        self.bwd_cache = None

    def update_weights(self, dLdW, dLdb):
        self.W += dLdW
        self.b += dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def get_gradients(self):
        return self.bwd_cache["dLdW"], self.bwd_cache["dLdb"]

    def forward(self, X, is_training=True):
        # flatten the input
        batch_size = X.shape[0]
        X_flattened = X.reshape(batch_size, -1)
        # apply the linear transformation
        out = np.dot(X_flattened, self.W.T) + self.b

        # save the forward cache for backprop
        if is_training:
            # store intermediate variables
            self.fwd_cache = {}
            self.fwd_cache["X"] = X_flattened

        return out

    def backprop(self, dLdy):
        # ensure is_training=True, i.e., the forward cache exists
        assert self.fwd_cache != None

        # load input data
        X = self.fwd_cache["X"]
        # compute gradient w.r.t. x
        dLdx = np.dot(dLdy, self.W)
        # compute gradient w.r.t. W
        dLdW = np.dot(dLdy.T, X)
        # compute gradient w.r.t. b
        dLdb = np.sum(dLdy, axis=0)

        # cache upstream gradients for weight updates
        self.bwd_cache = {}
        self.bwd_cache["dLdW"] = dLdW
        self.bwd_cache["dLdb"] = dLdb

        return dLdx, dLdW, dLdb

    
######################
## activation layer ##
##       ReLU       ##
######################

class nn_activation_layer_relu:
    def __init__(self):
        self.fwd_cache = None

    def forward(self, X, is_training=True):
        out = X.copy()  # deep copy
        out[out<0] = 0

        # save the forward cache for backprop
        if is_training:
            # store intermediate variables
            self.fwd_cache = {}
            self.fwd_cache["X"] = X

        return out

    def backprop(self, dLdy):
        # ensure is_training=True, i.e., the forward cache exists
        assert self.fwd_cache != None
        
        # load input data
        X = self.fwd_cache["X"]

        dLdx = dLdy.copy()
        dLdx[X<=0] = 0
        return dLdx
    

######################
## activation layer ##
##    Leaky ReLU    ##
######################

class nn_activation_layer_leaky_relu:
    def __init__(self, leak=0.01):
        self.fwd_cache = None
        self.leak = leak

    def forward(self, X, is_training=True):
        out = X.copy()  # deep copy
        out[out<0] *= self.leak

        # save the forward cache for backprop
        if is_training:
            # store intermediate variables
            self.fwd_cache = {}
            self.fwd_cache["X"] = X

        return out

    def backprop(self, dLdy):
        # ensure is_training=True, i.e., the forward cache exists
        assert self.fwd_cache != None
        
        # load input data
        X = self.fwd_cache["X"]

        dLdx = dLdy.copy()
        dLdx[X<=0] *= self.leak
        return dLdx


############################
## batch normalization 2d ##
############################

class nn_batchnorm_layer_2d:
    def __init__(self, num_features, momentum=0.9):
        self.momentum = momentum
        shape = (1, num_features, 1, 1)
        self.gamma = np.ones(shape)
        self.beta = np.zeros(shape)
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)
        self.epsilon = 1e-8
        self.fwd_cache = None
        self.bwd_cache = None

    def set_weights(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta

    def get_gradients(self):
        return self.bwd_cache["dLdgamma"], self.bwd_cache["dLdbeta"]

    def update_weights(self, dLdgamma, dLdbeta):
        self.gamma += dLdgamma
        self.beta += dLdbeta

    def forward(self, X, is_training=True):

        if is_training:
            # compute mean and var values of the input
            X_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
            X_var = np.var(X, axis=(0,2,3), keepdims=True)
            # update mean and var
            self.moving_mean = (1-self.momentum)*X_mean + self.momentum*self.moving_mean
            self.moving_var = (1-self.momentum)*X_var + self.momentum*self.moving_var
            # normalize
            X_hat = (X-X_mean) / np.sqrt(X_var+self.epsilon)
            # save forward cache
            self.fwd_cache = {}
            self.fwd_cache["X_mean_var_hat"] = (X, X_mean, X_var, X_hat)

        else:
            X_mean = self.moving_mean
            X_var = self.moving_var
            X_hat = (X-X_mean) / np.sqrt(X_var+self.epsilon)

        out = self.gamma * X_hat + self.beta
        return out

    def backprop(self, dLdy):
        epsilon = 1e-8
        # load input data and forward cache
        X, X_mean, X_var, X_hat = self.fwd_cache["X_mean_var_hat"]

        # input shape
        batch_size, _, inR, inC = X.shape

        # compute the gradients of gamma and beta
        dLdgamma = np.sum(dLdy * X_hat, axis=(0, 2, 3), keepdims=True)
        dLdbeta = np.sum(dLdy, axis=(0, 2, 3), keepdims=True)

        # compute gradient of X
        dLdX_hat = dLdy * self.gamma
        # dL/dX_var = dL/dX_hat * dX_hat/dX_var
        dX_var = np.sum(dLdX_hat * (X-X_mean)*-0.5*(X_var+epsilon)**(-1.5), axis=(0, 2, 3), keepdims=True)
        # dL/dX_mean = dL/dX_hat * dX_hat/dX_mean
        dX_mean = np.sum(dLdX_hat * -(X_var+epsilon)**(-0.5), axis=(0,2,3), keepdims=True) + dX_var * np.mean(-2.0 * (X-X_mean), axis=(0,2,3), keepdims=True)
        # dL/dX = dL/dX_hat * dX_hat/dX + dL/dX_var * dX_var/dX + dL/dX_mean * dX_mean/dX
        dLdx = dLdX_hat*(X_var+epsilon)**(-0.5) + dX_var*2.0*(X-X_mean)/batch_size/inR/inC + dX_mean/batch_size/inR/inC

        # cache upstream gradients for weight updates
        self.bwd_cache = {}
        self.bwd_cache["dLdgamma"] = dLdgamma
        self.bwd_cache["dLdbeta"] = dLdbeta

        return dLdx, dLdgamma, dLdbeta


############################
## batch normalization 1d ##
############################

class nn_batchnorm_layer_1d:
    def __init__(self, num_features, momentum=0.9):
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.moving_mean = np.zeros(num_features)
        self.moving_var = np.ones(num_features)
        self.epsilon = 1e-8
        self.fwd_cache = None
        self.bwd_cache = None

    def set_weights(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta

    def get_gradients(self):
        return self.bwd_cache["dLdgamma"], self.bwd_cache["dLdbeta"]

    def update_weights(self, dLdgamma, dLdbeta):
        self.gamma += dLdgamma
        self.beta += dLdbeta

    def forward(self, X, is_training=True):

        if is_training:
            # compute mean and var values of the input
            X_mean = np.mean(X, axis=0, keepdims=True)
            X_var = np.var(X, axis=0, keepdims=True)
            # update mean and var
            self.moving_mean = (1-self.momentum)*X_mean + self.momentum*self.moving_mean
            self.moving_var = (1-self.momentum)*X_var + self.momentum*self.moving_var
            # normalize
            X_hat = (X-X_mean) / np.sqrt(X_var+self.epsilon)
            # save forward cache
            self.fwd_cache = {}
            self.fwd_cache["X_mean_var_hat"] = (X, X_mean, X_var, X_hat)

        else:
            X_mean = self.moving_mean
            X_var = self.moving_var
            X_hat = (X-X_mean) / np.sqrt(X_var+self.epsilon)

        out = self.gamma * X_hat + self.beta
        return out

    def backprop(self, dLdy):
        epsilon = 1e-8
        # load input data and forward cache
        X, X_mean, X_var, X_hat = self.fwd_cache["X_mean_var_hat"]

        # input shape
        batch_size, num_features = X.shape

        # compute the gradients of gamma and beta
        dLdgamma = np.sum(dLdy * X_hat, axis=0)
        dLdbeta = np.sum(dLdy, axis=0)

        # compute gradient of X
        dLdX_hat = dLdy * self.gamma
        # dL/dX_var = dL/dX_hat * dX_hat/dX_var
        dX_var = np.sum(dLdX_hat * (X-X_mean)*-0.5*(X_var+epsilon)**(-1.5), axis=(0), keepdims=True)
        # dL/dX_mean = dL/dX_hat * dX_hat/dX_mean
        dX_mean = np.sum(dLdX_hat * -(X_var+epsilon)**(-0.5), axis=0, keepdims=True) + dX_var * np.mean(-2.0 * (X-X_mean), axis=0, keepdims=True)
        # dL/dX = dL/dX_hat * dX_hat/dX + dL/dX_var * dX_var/dX + dL/dX_mean * dX_mean/dX
        dLdx = dLdX_hat*(X_var+epsilon)**(-0.5) + dX_var*2.0*(X-X_mean)/batch_size + dX_mean/batch_size

        # cache upstream gradients for weight updates
        self.bwd_cache = {}
        self.bwd_cache["dLdgamma"] = dLdgamma
        self.bwd_cache["dLdbeta"] = dLdbeta

        return dLdx, dLdgamma, dLdbeta


###################
## softmax layer ##
###################

class nn_softmax_layer:
    def __init__(self):
        self.fwd_cache = None

    def forward(self, X, is_training=True):
        # subtract max for numerical stability
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        # softmax formula
        out = e_x / np.sum(e_x, axis=-1, keepdims=True)

        # save the forward cache for backprop
        if is_training:
            # store intermediate variables
            self.fwd_cache = {}
            self.fwd_cache["X"] = X

        return out

    def backprop(self, dLdy):
        # ensure is_training=True, i.e., the forward cache exists
        assert self.fwd_cache != None

        return dLdy


#########################
## cross entropy layer ##
#########################

class nn_cross_entropy_layer:
    def __init__(self):
        self.fwd_cache = None

    def forward(self, X, y, is_training=True):
        epsilon = 1e-5
        # compute cross entropy
        log_likelihood = -np.log(X[range(y.shape[0]), y] + epsilon)
        loss = np.sum(log_likelihood) / y.shape[0]

        # save the forward cache for backprop
        if is_training:
            # store intermediate variables
            self.fwd_cache = {}
            self.fwd_cache["X"] = X

        return loss

    def backprop(self, y):
        # ensure is_training=True, i.e., the forward cache exists
        assert self.fwd_cache != None

        # load input data
        X = self.fwd_cache["X"]
        # the gradient is simply the softmax output - y
        num_examples = X.shape[0]
        dx = X.copy()
        dx[range(num_examples), y] -= 1
        dx /= num_examples
        return dx