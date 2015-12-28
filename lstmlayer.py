import numpy as np
import numpy.random as rd

from activations import *

class LSTMGateLayer(object):
    def __init__(self):
        # Do nothing.
        pass

    def initialize(self, input_dim, hidden_dim):
        # U connects h(t-1), W connects x(t) and V connects c(t-1)
        self.U = rd.randn(hidden_dim, hidden_dim) * 0.001
        self.W = rd.randn(input_dim, hidden_dim) * 0.001
        self.V = rd.randn(hidden_dim, hidden_dim) * 0.001
        self.B = rd.randn(hidden_dim) * 0.001

        self.U_update = np.zeros_like(self.U)
        self.W_update = np.zeros_like(self.W)
        self.V_update = np.zeros_like(self.V)
        self.B_update = np.zeros_like(self.B)

        self.activation = create_activation('sigmod')

    def copy(self):
        cp = LSTMGateLayer()
        cp.U = self.U
        cp.W = self.W
        cp.V = self.V
        cp.B = self.B
        cp.U_update = self.U_update
        cp.W_update = self.W_update
        cp.V_update = self.V_update
        cp.B_update = self.B_update
        cp.activation = self.activation
        return cp

    def forward(self, x, pre_h, pre_c):
        self.x = x
        self.pre_h = pre_h
        self.pre_c = pre_c
        net = self.W.T.dot(x) + self.B
        if pre_h != None:
            net += self.U.T.dot(pre_h)
        if pre_c != None:
            net += self.V.T.dot(pre_c)
        self.output = self.activation.fn(net)
        return self.output

    def backward(self, delta):
        delta_ = self.activation.deriv(self.output) * delta
        self.W_update += np.outer(self.x, delta_)
        self.B_update += delta_
        delta_pre_h, delta_pre_c = None, None
        if self.pre_h != None:
            self.U_update += np.outer(self.pre_h, delta_)
            delta_pre_h = delta_.dot(self.U.T)
        if self.pre_c != None:
            self.V_update += np.outer(self.pre_c, delta_)
            delta_pre_c = delta_.dot(self.V.T)
        return (delta_pre_h, delta_pre_c)

    def update_weights(self, lamd):
        self.U += self.U_update * lamd
        self.W += self.W_update * lamd
        self.V += self.V_update * lamd
        self.B += self.B_update * lamd
        self.U_update *= 0
        self.W_update *= 0
        self.V_update *= 0
        self.B_update *= 0

    def clean(self):
        self.__delattr__('U_update')
        self.__delattr__('W_update')
        self.__delattr__('V_update')
        self.__delattr__('B_update')

class LSTMCellLayer(object):
    def __init__(self):
        # Do nothing.
        pass

    def initialize(self, input_dim, hidden_dim, activation):
        # U connects h(t-1), W connects x(t) and V connects c(t-1)
        self.U = rd.randn(hidden_dim, hidden_dim) * 0.001
        self.W = rd.randn(input_dim, hidden_dim) * 0.001
        self.B = rd.randn(hidden_dim) * 0.001

        self.U_update = np.zeros_like(self.U)
        self.W_update = np.zeros_like(self.W)
        self.B_update = np.zeros_like(self.B)

        self.activation = activation

    def copy(self):
        cp = LSTMCellLayer()
        cp.U = self.U
        cp.W = self.W
        cp.B = self.B
        cp.U_update = self.U_update
        cp.W_update = self.W_update
        cp.B_update = self.B_update
        cp.activation = self.activation
        return cp

    def forward(self, x, pre_h):
        self.x = x
        self.pre_h = pre_h
        if pre_h != None:
            self.output = self.activation.fn(self.W.T.dot(x) + \
                          self.U.T.dot(pre_h) + self.B)
        else:
            self.output = self.activation.fn(self.W.T.dot(x) + self.B)
        return self.output

    def backward(self, delta):
        delta_ = self.activation.deriv(self.output) * delta
        self.W_update += np.outer(self.x, delta_)
        self.B_update += delta_
        delta_pre_h = None
        if self.pre_h != None:
            self.U_update += np.outer(self.pre_h, delta_)
            delta_pre_h = delta_.dot(self.U.T)
        return delta_pre_h

    def update_weights(self, lamd):
        self.U += self.U_update * lamd
        self.W += self.W_update * lamd
        self.B += self.B_update * lamd
        self.U_update *= 0
        self.W_update *= 0
        self.B_update *= 0

    def clean(self):
        self.__delattr__('U_update')
        self.__delattr__('W_update')
        self.__delattr__('B_update')

class LSTMLayer(object):
    def __init__(self):
        # Do nothing. Let def initialize to do that.
        pass

    def initialize(self, input_dim, hidden_dim, activation):
        self.input_gate = LSTMGateLayer()
        self.forget_gate = LSTMGateLayer()
        self.output_gate = LSTMGateLayer()
        self.cell = LSTMCellLayer()
        self.input_gate.initialize(input_dim, hidden_dim)
        self.forget_gate.initialize(input_dim, hidden_dim)
        self.output_gate.initialize(input_dim, hidden_dim)
        self.cell.initialize(input_dim, hidden_dim, activation)

        self.activation = activation


    def copy(self):
        cp = LSTMLayer()
        cp.input_gate = self.input_gate.copy()
        cp.forget_gate = self.forget_gate.copy()
        cp.output_gate = self.output_gate.copy()
        cp.cell = self.cell.copy()
        cp.activation = self.activation
        return cp

    def forward(self, x, pre_h, pre_c):
        self.pre_h = pre_h
        self.pre_c = pre_c
        self.ig = self.input_gate.forward(x, pre_h, pre_c)
        self.fg = self.forget_gate.forward(x, pre_h, pre_c)
        self.c_ = self.cell.forward(x, pre_h)
        self.c = self.ig * self.c_
        if pre_c != None:
            self.c += self.fg * pre_c
        self.og = self.output_gate.forward(x, pre_h, self.c)
        self.c_act = self.activation.fn(self.c)
        self.output = self.c_act * self.og
        return (self.output, self.c)

    def backward(self, future_delta_h, future_delta_c, delta):
        delta_ = delta
        if future_delta_h != None:
            delta_ += future_delta_h
        # Errors to forget gate
        delta_o = self.c_act * delta_
        delta_pre_h_o, delta_c_o = self.output_gate.backward(delta_o)
        
        delta_c_act = self.og * delta_
        delta_c = self.activation.deriv(self.c_act) * delta_c_act
        delta_c += delta_c_o
        if future_delta_c != None:
            delta_c += future_delta_c
        delta_i = self.c_ * delta_c
        delta_pre_h_i, delta_pre_c_i = self.input_gate.backward(delta_i)
        
        delta_c_ = self.ig * delta_c
        delta_pre_h_c_ = self.cell.backward(delta_c_)

        delta_f = delta_pre_h_f = delta_pre_c_f = delta_pre_c = None
        if self.pre_c != None:
            delta_f = self.pre_c * delta_c
            delta_pre_h_f, delta_pre_c_f = self.forget_gate.backward(delta_f)

            delta_pre_c = self.fg * delta_c

        def sum_deltas(deltas):
            deltas = [delta for delta in deltas if delta != None]
            if len(deltas) == 0:
                return None
            return sum(deltas)

        delta_pre_c = sum_deltas([delta_pre_c, delta_pre_c_i,\
                                 delta_pre_c_f])
        delta_pre_h = sum_deltas([delta_pre_h_o, delta_pre_h_i,\
                           delta_pre_h_c_, delta_pre_h_f])
        return (delta_pre_h, delta_pre_c)

    def update_weights(self, lamd):
        self.input_gate.update_weights(lamd)
        self.output_gate.update_weights(lamd)
        self.forget_gate.update_weights(lamd)
        self.cell.update_weights(lamd)

    def clean(self):
        self.input_gate.clean()
        self.output_gate.clean()
        self.forget_gate.clean()
        self.cell.clean()