'''A simple RNN.'''

__author__ = 'Chunqi Wang'
__date__ = 'November,2015'

import random
from itertools import *
import numpy as np
import numpy.linalg as la
import numpy.random as rd

# Activation functions.
class Linear():
    def fn(self, x):
        return x
    def deriv(self, z):
        return 1
class Sigmod():
    def fn(self, x):
        if x < 50:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return self.fn(50)
    def deriv(self, z):
        return z * (1 - z)
class Tanh():
    def fn(self, x):
        if x < 50:
            return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))
        else:
            return self.fn(50)
    def deriv(self, z):
        return (1.0 + z) * (1.0 - z) / 2
class ReLU():
    def fn(self, x):
        return (x + np.abs(x)) / 2.0
    def deriv(self, z):
        return (z > 0).astype(np.float)

activations = {'tanh': Tanh(), 'sigmod': Sigmod(), 'linear': Linear(), 'ReLU': ReLU()}

class RNN():
    def __init__(self, hidden_dim=32, f_hidden='tanh', max_seq_len=30, lamd=0.1, iterations=1000, debug=False, showinfo = 100):
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.debug = debug

        self.f_hidden = activations[f_hidden]
        self.f_output = activations['linear']

        self.showinfo = showinfo
        self.iterations = iterations
        self.lamd = lamd
        self.eta = 0.0001

    def fit(self, Seq_input, Seq_target):
        Seq_input = [seq_input.reshape((seq_input.shape[0], 1)) \
                     for seq_input in Seq_input if seq_input.ndim == 1]
        Seq_target = [seq_target.reshape((seq_target.shape[0], 1)) \
                     for seq_target in Seq_target if seq_target.ndim == 1]

        self.input_dim = Seq_input[0].shape[1]
        self.ouput_dim = Seq_target[0].shape[1]

        # Network parameters.
        self.W_i = rd.randn(self.input_dim, self.hidden_dim) / 1000
        self.W_h = rd.randn(self.hidden_dim, self.hidden_dim) / 1000
        self.B_i = rd.randn(self.hidden_dim) / 1000
        self.W_o = rd.randn(self.hidden_dim, self.ouput_dim) / 1000
        self.B_o = rd.randn(self.ouput_dim) / 1000

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
                index = rd.randint(len(Seq_input)-1)
                print 'iterations: ',iteration , ' object function value: ', \
                      self._objfunc_value(Seq_input[index], Seq_target[index])
            for input_seq, output_seq in izip(Seq_input, Seq_target):
                seq_len = len(input_seq)
                # Forward.
                for i in xrange(seq_len):
                    x = input_seq[i]
                    Z_h[i] = self.f_hidden.fn((x.dot(self.W_i) + Z_h[i - 1].dot(self.W_h) + self.B_i))
                    Z_o[i] = self.f_output.fn(Z_h[i].dot(self.W_o) + self.B_o)
                # Backward.
                for i in xrange(seq_len - 1, -1, -1):
                    x, y = input_seq[i], output_seq[i]
                    Delta_o[i] = mse(Z_o[i], y) * self.f_output.deriv(Z_o[i])
                    Delta_h[i] = (Delta_o[i].dot(self.W_o.T) + Delta_h[i + 1].dot(self.W_h.T)) * self.f_hidden.deriv(Z_h[i])
                    # Calculate updates.
                    W_o_update -= np.outer(Z_h[i], Delta_o[i])
                    B_o_update -= Delta_o[i]
                    W_i_update -= np.outer(x, Delta_h[i])
                    B_i_update -= Delta_h[i]
                    W_h_update -= np.outer(Z_h[i - 1], Delta_h[i])
                
                if self.debug:
                    # Gradient checking.
                    g = self._gradient(input_seq, output_seq, self.W_h[0:1, 0:1])
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
        result = []
        for input_seq in Seq_input:
            if input_seq.ndim == 1:
                input_seq = input_seq.reshape((input_seq.shape[0], 1))
            pre_z_h = np.zeros(self.hidden_dim)
            seq_len = len(input_seq)
            Z_o = []
            for i in xrange(seq_len):
                x = input_seq[i]
                z_h = self.f_hidden.fn(x.dot(self.W_i) + pre_z_h.dot(self.W_h) + self.B_i)
                Z_o.append(self.f_output.fn(z_h.dot(self.W_o) + self.B_o))
                pre_z_h = z_h
            Z_o = np.array(Z_o)
            if Z_o.shape[1] == 1:
                Z_o = Z_o.reshape((seq_len))
            result.append(Z_o)
        return result

    def average_error(self, Seq_input, Seq_target):
        error = 0
        for input_seq, output_seq in izip(Seq_input, Seq_target):
            error += self._objfunc_value(input_seq, output_seq)
        return error / len(Seq_input)

    def _objfunc_value(self, input_seq, output_seq):
        if input_seq.ndim == 1:
            input_seq = input_seq.reshape((input_seq.shape[0], 1))
        if output_seq.ndim == 1:
            output_seq = output_seq.reshape((output_seq.shape[0], 1))
        seq_len = len(input_seq)
        pre_z_h = np.zeros(self.hidden_dim)
        error = 0
        for i in xrange(seq_len):
            x = input_seq[i]
            z_h = self.f_hidden.fn(x.dot(self.W_i) + pre_z_h.dot(self.W_h) + self.B_i)
            z_o = self.f_output.fn(z_h.dot(self.W_o) + self.B_o)
            pre_z_h = z_h
            error += (z_o - output_seq[i])[0]**2
        return error / 2.0

    def _gradient(self, input_seq, output_seq, w):
        v1 = self._objfunc_value(input_seq, output_seq)
        delta = 10**(-5)
        w += delta
        v2 = self._objfunc_value(input_seq, output_seq)
        w -= delta
        return (v2 - v1) / delta