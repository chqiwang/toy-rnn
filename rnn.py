'''A simple RNN.'''

__author__ = 'Chunqi Wang'
__date__ = 'November,2015'

import random
from itertools import izip
import numpy as np
import numpy.linalg as la

from activations import *
from layers import *

class RNN(object):
    def __init__(self, hidden_dim=32, f_hidden='sigmod', lamd=0.1,\
                 iterations=1000, debug=False, showinfo = 100):
        self.hidden_dim = hidden_dim

        self.debug = debug

        self.f_hidden = create_activation(f_hidden)
        self.f_output = create_activation('linear')

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

        max_seq_len = 0
        for seq_input in Seq_input:
            max_seq_len = max(max_seq_len, len(seq_input))

        # Network layers.
        self.output_layer = OutputLayer()
        self.output_layer.initialize(self.hidden_dim, self.ouput_dim, self.f_output)
        output_layers = [self.output_layer.copy() for i in xrange(max_seq_len)]
        self.hidden_layer = HiddenLayer()
        self.hidden_layer.initialize(self.input_dim, self.hidden_dim, self.output_layer.W, self.f_hidden)
        hidden_layers = [self.hidden_layer.copy() for i in xrange(max_seq_len)]

        mse = lambda z, y: y - z

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
                v = hidden_layers[0].forward(input_seq[0], None)
                output_layers[0].forward(v)
                for i in xrange(1, seq_len):
                    v = hidden_layers[i].forward(input_seq[i], hidden_layers[i-1].output)
                    output_layers[i].forward(v)
                    
                # Backward.
                output_layers[-1].backward(mse(output_layers[-1].output, output_seq[-1]))
                hidden_layers[-1].backward(None, output_layers[-1].delta)
                for i in xrange(seq_len - 2, -1, -1):
                    output_layers[i].backward(mse(output_layers[i].output, output_seq[i]))
                    hidden_layers[i].backward(hidden_layers[i+1].delta, output_layers[i].delta)
                
                if self.debug:
                    # Gradient checking.
                    g = self._gradient(input_seq, output_seq, self.hidden_layer.W_i[0:1, 0:1])
                    if abs(g + self.hidden_layer.W_i_update[0, 0]) >= 10**(-5):
                        print 'gradient check error.', g, - self.hidden_layer.W_i_update[0, 0]
                    else:
                        print 'gradient check right.', g, - self.hidden_layer.W_i_update[0, 0]

                # Update all weights.
                self.output_layer.update_weights(self.lamd)
                self.hidden_layer.update_weights(self.lamd)
        # Delete non-usage memories.
        self.hidden_layer.clean()
        self.output_layer.clean()

    def predict(self, Seq_input):
        result = []
        for input_seq in Seq_input:
            if input_seq.ndim == 1:
                input_seq = input_seq.reshape((input_seq.shape[0], 1))
            output = []
            v = None
            for x in input_seq:
                v = self.hidden_layer.forward(x, v)
                w = self.output_layer.forward(v)
                output.append(w)
            output = np.array(output)
            if output.shape[1] == 1:
                output = output.reshape((output.shape[0]))
            result.append(output)
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
        v = None
        error = 0.0
        for x, y in izip(input_seq, output_seq):
            v = self.hidden_layer.forward(x, v)
            w = self.output_layer.forward(v)
            error += (w - y)[0]**2
        return error / 2.0

    def _gradient(self, input_seq, output_seq, w):
        v1 = self._objfunc_value(input_seq, output_seq)
        delta = 10**(-5)
        w += delta
        v2 = self._objfunc_value(input_seq, output_seq)
        w -= delta
        return (v2 - v1) / delta