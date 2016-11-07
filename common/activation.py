''' Number of activation functions '''
import autograd.numpy as np

identity = lambda x: x

tanh = lambda x: np.tanh(x)

sigmoid = lambda x: 0.5*(np.tanh(x) + 1.0)

rbf = lambda x: np.exp(-x**2)

relu = lambda x: np.maximum(0, x)
