import numpy as np

# Activation functions.
class Linear(object):
    def fn(self, x):
        return x
    def deriv(self, z):
        return 1
class Sigmod(object):
    def fn(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def deriv(self, z):
        return z * (1 - z)
class Tanh(object):
    def fn(self, x):
        return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))
    def deriv(self, z):
        return (1.0 + z) * (1.0 - z) / 2
class ReLU(object):
    def fn(self, x):
        return (x + np.abs(x)) / 2.0
    def deriv(self, z):
        return (z > 0).astype(np.float)

def create_activation(name):
    name = name.lower()
    if name == 'tanh':
        return Tanh()
    if name == 'sigmod':
        return Sigmod()
    if name == 'linear':
        return Linear()
    if name == 'relu':
        return ReLU()