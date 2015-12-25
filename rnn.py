'''A simple RNN.'''

__author__ = 'Chunqi Wang'
__date__ = 'November,2015'

import random
from itertools import *
import numpy as np
import numpy.linalg as la
import numpy.random as rd

# Activation functions.
linear = lambda x: x
d_linear = lambda z: 1.0
sigmod = lambda x: 1.0 / (1.0 + np.exp(-x))
d_sigmod = lambda z: z * (1.0 - z)
tanh = lambda x: (np.exp(x) - 1.0) / (np.exp(x) + 1.0)
d_tanh = lambda z: 1.0 / 2.0 * (1.0 + z) * (1.0 - z)
softmax = lambda x: np.exp(x) / sum(np.exp(x))
d_softmax = lambda z: z * (1.0 - z)
ReLu = lambda x: (x + np.abs(x)) / 2.0
d_ReLu = lambda z: (z > 0).astype(np.float)

activations = {'tanh': tanh, 'linear': linear, 'Relu': ReLu}
d_activations = {'tanh': d_tanh, 'linear': d_linear, 'Relu': d_ReLu}

class RNN():
    def __init__(self, hidden_dim=32, max_seq_len=30, lamd=0.1, iterations=1000, debug=False, showinfo = 100):
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.debug = debug

        self.f_hidden = 'tanh'
        self.f_output = 'linear'

        self.showinfo = showinfo
        self.iterations = iterations
        self.lamd = lamd
        self.eta = 0.0001

    def fit(self, Seq_input, Seq_target):

        if Seq_input.ndim == 2:
            Seq_input = Seq_input.reshape((Seq_input.shape[0], Seq_input.shape[1], 1))
        if Seq_target.ndim == 2:
            Seq_target = Seq_target.reshape((Seq_target.shape[0], Seq_target.shape[1], 1))

        self.input_dim = Seq_input.shape[2]
        self.ouput_dim = Seq_target.shape[2]

        # Network parameters.
        self.W_i = rd.randn(self.input_dim, self.hidden_dim)
        self.W_h = rd.randn(self.hidden_dim, self.hidden_dim)
        self.B_i = rd.randn(self.hidden_dim)
        self.W_o = rd.randn(self.hidden_dim, self.ouput_dim)
        self.B_o = rd.randn(self.ouput_dim)

        mse = lambda z, y: z - y

        W_i_update = np.zeros_like(self.W_i)
        W_h_update = np.zeros_like(self.W_h)
        B_i_update = np.zeros_like(self.B_i)
        W_o_update = np.zeros_like(self.W_o)
        B_o_update = np.zeros_like(self.B_o)

        Z_h = np.zeros((self.max_seq_len + 1, self.hidden_dim))
        Z_o = np.zeros((self.max_seq_len, self.ouput_dim))
        Delta_h = np.zeros_like(Z_h)
        Delta_o = np.zeros_like(Z_o)

        Weights = [self.W_i, self.W_h, self.B_i, self.W_o, self.B_o]
        Weights_update = [W_i_update, W_h_update, B_i_update, W_o_update, B_o_update]
        temporary = [W_i_update, W_h_update, B_i_update, W_o_update, B_o_update, Z_h, Z_o, Delta_h, Delta_o]

        # Train the network by BP algorithm.
        for iteration in xrange(self.iterations):
            # Output error informations for one random sample.
            if iteration % self.showinfo == 0:
                index = rd.randint(len(Seq_input-1))
                print 'iterations: ',iteration , ' object function value: ', \
                      self.object_function_value(Seq_input[index], Seq_target[index])
            for input_seq, output_seq in izip(Seq_input, Seq_target):
                seq_len = len(input_seq)
                # Forward.
                for i in xrange(seq_len):
                    x = input_seq[i]
                    Z_h[i] = activations[self.f_hidden](x.dot(self.W_i) + Z_h[i - 1].dot(self.W_h) + self.B_i)
                    Z_o[i] = activations[self.f_output](Z_h[i].dot(self.W_o) + self.B_o)
                # Backward.
                for i in xrange(seq_len - 1, -1, -1):
                    x, y = input_seq[i], output_seq[i]
                    Delta_o[i] = mse(Z_o[i], y) * d_activations[self.f_output](Z_o[i])
                    Delta_h[i] = (Delta_o[i].dot(self.W_o.T) + Delta_h[i + 1].dot(self.W_h.T)) * d_activations[self.f_hidden](Z_h[i])
                    # Calculate updates.
                    W_o_update -= np.outer(Z_h[i], Delta_o[i])
                    B_o_update -= Delta_o[i]
                    W_i_update -= np.outer(x, Delta_h[i])
                    B_i_update -= Delta_h[i]
                    W_h_update -= np.outer(Z_h[i - 1], Delta_h[i])
                
                if self.debug:
                    # Gradient checking.
                    g = self.gradient(input_seq, output_seq, self.W_h[0:1, 0:1])
                    if abs(g + W_h_update[0, 0]) >= 10**(-5):
                        print 'gradient check error.', g, - W_h_update[0, 0]
                    else:
                        print 'gradient check right.', g, - W_h_update[0, 0]

                # Update all weights.
                for w, w_u in izip(Weights, Weights_update):
                    w += self.lamd * w_u - self.eta * w
                # Reset memories for reuse.
                for t in temporary: t *= 0

    def predict(self, Seq_input):
        if Seq_input.ndim == 2:
            Seq_input = Seq_input.reshape((Seq_input.shape[0], Seq_input.shape[1], 1))
        result = []
        for input_seq in Seq_input:
            pre_z_h = np.zeros(self.hidden_dim)
            seq_len = len(input_seq)
            Z_o = []
            for i in xrange(seq_len):
                x = input_seq[i]
                z_h = activations[self.f_hidden](x.dot(self.W_i) + pre_z_h.dot(self.W_h) + self.B_i)
                Z_o.append(activations[self.f_output](z_h.dot(self.W_o) + self.B_o))
                pre_z_h = z_h
            result.append(Z_o)
        result = np.array(result)
        if result.shape[2] == 1:
            result = result.reshape((result.shape[0], result.shape[1]))
        return result

    def object_function_value(self, input_seq, output_seq):
        if input_seq.ndim == 1:
            input_seq = input_seq.reshape((input_seq.shape[0], 1))
        if output_seq.ndim == 1:
            output_seq = output_seq.reshape((output_seq.shape[0], 1))
        seq_len = len(input_seq)
        pre_z_h = np.zeros(self.hidden_dim)
        error = 0
        for i in xrange(seq_len):
            x = input_seq[i]
            z_h = activations[self.f_hidden](x.dot(self.W_i) + pre_z_h.dot(self.W_h) + self.B_i)
            z_o = activations[self.f_output](z_h.dot(self.W_o) + self.B_o)
            pre_z_h = z_h
            error += (z_o - output_seq[i])[0]**2
        return error / 2.0

    def gradient(self, input_seq, output_seq, w):
        v1 = self.object_function_value(input_seq, output_seq)
        delta = 10**(-5)
        w += delta
        v2 = self.object_function_value(input_seq, output_seq)
        w -= delta
        return (v2 - v1) / delta