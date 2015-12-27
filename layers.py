import numpy as np
import numpy.random as rd

class HiddenLayer(object):
    def __init__(self):
        # Do nothing. Let def initialize to do that.
        pass

    def initialize(self, input_dim, hidden_dim, W_output, activation):
        self.W_i = rd.randn(input_dim, hidden_dim) * 0.001
        self.W_h = rd.randn(hidden_dim, hidden_dim) * 0.001
        self.B = rd.randn(hidden_dim) * 0.001
        self.activation = activation
        self.W_i_update = np.zeros_like(self.W_i)
        self.W_h_update = np.zeros_like(self.W_h)
        self.B_update = np.zeros_like(self.B)
        self.W_output = W_output

    def copy(self):
        cp = HiddenLayer()
        cp.W_i = self.W_i
        cp.W_h = self.W_h
        cp.B = self.B
        cp.activation = self.activation
        cp.W_i_update = self.W_i_update
        cp.W_h_update = self.W_h_update
        cp.B_update = self.B_update
        cp.W_output = self.W_output
        return cp

    def forward(self, x, h_pre):
        self.x = x
        self.h_pre = h_pre
        if h_pre != None:
            self.output = self.activation.fn(self.W_i.T.dot(x) + self.W_h.T.dot(h_pre) + self.B)
        else:
            self.output = self.activation.fn(self.W_i.T.dot(x) + self.B)
        return self.output

    def backward(self, pre_hidden_delta, delta):
        if pre_hidden_delta != None:
            self.delta = self.activation.deriv(self.output) * \
                         (delta.dot(self.W_output.T) + pre_hidden_delta.dot(self.W_h.T))
        else:
            self.delta = self.activation.deriv(self.output) * (delta.dot(self.W_output.T))
        self.W_i_update += np.outer(self.x, self.delta)
        if self.h_pre != None:
            self.W_h_update += np.outer(self.h_pre, self.delta)
        self.B_update += self.delta
        return self.delta

    def update_weights(self, lamd):
        self.W_i += self.W_i_update * lamd
        self.W_h += self.W_h_update * lamd
        self.B += self.B_update * lamd
        self.W_i_update *= 0
        self.W_h_update *= 0
        self.B_update *= 0

    def clean(self):
        self.__delattr__('W_i_update')
        self.__delattr__('W_h_update')
        self.__delattr__('B_update')
        self.__delattr__('W_output')

class OutputLayer(object):
    def __init__(self):
        # Do nothing. Let def initialize to do that.
        pass

    def initialize(self, hidden_dim, output_dim, activation):
        self.W = rd.randn(hidden_dim, output_dim) * 0.001
        self.B = rd.randn(output_dim) *0.001
        self.activation = activation
        self.W_update = np.zeros_like(self.W)
        self.B_update = np.zeros_like(self.B)

    def copy(self):
        cp = OutputLayer()
        cp.W = self.W
        cp.B = self.B
        cp.activation = self.activation
        cp.W_update = self.W_update
        cp.B_update = self.B_update
        return cp

    def forward(self, x):
        self.x = x
        self.output = self.activation.fn(self.W.T.dot(x) + self.B)
        return self.output

    def backward(self, delta):
        self.delta = self.activation.deriv(self.output) * delta
        self.W_update += np.outer(self.x, self.delta)
        return self.delta

    def update_weights(self, lamd):
        self.W += self.W_update * lamd
        self.B += self.B_update * lamd
        self.W_update *= 0
        self.B_update *= 0

    def clean(self):
        self.__delattr__('W_update')
        self.__delattr__('B_update')