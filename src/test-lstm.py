''' This is a test program.'''

import cPickle as pickle
import numpy as np, scipy as sp
import numpy.random as rd
from lstm import RNN

n, seq_len = 100, 10

def train():
    Seq_input = rd.rand(n, seq_len)
    Seq_output = np.zeros_like(Seq_input)
    Seq_output[:,0] = Seq_input[:,0]
    for i in xrange(1, seq_len):
        Seq_output[:,i] += Seq_output[:,i-1] + Seq_input[:,i]
    
    rnn = RNN(lamd=0.1, hidden_dim=8, f_hidden='Tanh', iterations=100, debug=False, showinfo=10)
    rnn.fit(Seq_input, Seq_output)

    with open('RNN-model.pkl', 'w+') as f:
        pickle.dump(rnn, f)

def test():
    # seq_len = 10
    Seq_input_test = rd.rand(n, seq_len) * 1
    Seq_output_test = np.zeros_like(Seq_input_test)
    Seq_output_test[:,0] = Seq_input_test[:,0]
    for i in xrange(1, seq_len):
        Seq_output_test[:,i] += Seq_output_test[:,i-1] + Seq_input_test[:,i]
    with open('RNN-model.pkl') as f:
        rnn = pickle.load(f)
    print 'average error:', rnn.average_error(Seq_input_test, Seq_output_test)
    print 'sample:'
    print 'target:', Seq_output_test[0]
    print 'predict', rnn.predict(Seq_input_test[:1])[0]

if __name__ == '__main__':
    # do test.
    train()
    test()