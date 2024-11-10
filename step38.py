import numpy as np
from dezero import Variable

x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
y = x.reshape(2, 3)
print(y)


x = Variable(np.random.rand(2, 3))
y = x.transpose()
print(x)
print(y)

