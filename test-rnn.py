''' This is a test program.'''

import cPickle as pickle
import numpy as np, scipy as sp
import numpy.random as rd
from rnn import RNN

n, seq_len = 1000, 10

def train():
    Seq_input = rd.rand(n, seq_len) * 10
    Seq_output = np.zeros_like(Seq_input)
    Seq_output[:,0] = Seq_input[:,0]
    for i in xrange(1, seq_len):
        Seq_output[:,i] += Seq_output[:,i-1] + Seq_input[:,i]
    
    rnn = RNN(lamd=0.001, hidden_dim=64, max_seq_len=seq_len, iterations=100, debug=False, showinfo=10)
    rnn.fit(Seq_input, Seq_output)

    with open('RNN-model.pkl', 'w+') as f:
        pickle.dump(rnn, f)

def test:
    with open('RNN-model.pkl') as f:
        rnn = pickle.load(f)
    Seq_input_test = np.array([range(seq_len)])
    print rnn.predict(Seq_input_test)[0]

if __name__ == '__main__':
    # do something...