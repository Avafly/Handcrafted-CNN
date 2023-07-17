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

    def get_gradient(self):
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

    def get_gradient(self):
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

class nn_activation_layer:
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
        eps = 1e-10
        # compute cross entropy
        log_likelihood = -np.log(X[range(y.shape[0]), y] + eps)
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