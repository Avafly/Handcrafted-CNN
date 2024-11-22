import json
import numpy as np
import nn_layers as nnl

class nn_classifier:
    def __init__(self, config_path, rmsprop_beta=0.9, lr=1.0e-2):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.layers = []
        for layer_cfg in config["layers"]:
            layer = layer_creator.create_layer(layer_cfg["type"], layer_cfg["params"])
            self.layers.append(layer)
        
        self.rmsprop_beta = rmsprop_beta
        self.lr = lr
        self.epsilon = 1e-5
        self.is_first_update = True
        self.fwd_cache = None

    def forward(self, X, y, is_training=True):
        outputs = X
        for layer in self.layers[:-1]:
            outputs = layer.forward(outputs, is_training)
        loss = self.layers[-1].forward(outputs, y, is_training)
        if is_training:
            self.fwd_cache = (X, y)
        return outputs, loss

    def backprop(self):
        dLdy = self.layers[-1].backprop(self.fwd_cache[1])
        for layer in reversed(self.layers[:-1]):
            dLdy = layer.backprop(dLdy)

    def update_weights(self):
        beta, lr, epsilon = self.rmsprop_beta, self.lr, self.epsilon
        if self.is_first_update:
            self.velocity = {id(layer): {"w": 0, "b": 0} for layer in self.layers if hasattr(layer, "get_gradients")}
            self.is_first_update = False
        
        for layer in self.layers:
            if hasattr(layer, "get_gradients"):
                dLdW, dLdb = layer.get_gradients()
                v = self.velocity[id(layer)]
                v["w"] = beta * v["w"] + (1 - beta) * dLdW ** 2
                v["b"] = beta * v["b"] + (1 - beta) * dLdb ** 2
                layer.update_weights(
                    -lr * dLdW / (np.sqrt(v["w"]) + epsilon),
                    -lr * dLdb / (np.sqrt(v["b"]) + epsilon)
                )


class layer_creator:
    @staticmethod
    def create_layer(layer_type, params):
        if layer_type == "conv":
            return nnl.nn_convolutional_layer(**params)
        elif layer_type == "relu":
            return nnl.nn_activation_layer_relu()
        elif layer_type == "leakyrelu":
            return nnl.nn_activation_layer_leaky_relu()
        elif layer_type == "maxpool":
            return nnl.nn_max_pooling_layer(**params)
        elif layer_type == "avgpool":
            return nnl.nn_avg_pooling_layer(**params)
        elif layer_type == "fc":
            return nnl.nn_fc_layer(**params)
        elif layer_type == "bn2d":
            return nnl.nn_batchnorm_layer_2d(**params)
        elif layer_type == "bn1d":
            return nnl.nn_batchnorm_layer_1d(**params)
        elif layer_type == "softmax":
            return nnl.nn_softmax_layer()
        elif layer_type == "cross_entropy":
            return nnl.nn_cross_entropy_layer()
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")