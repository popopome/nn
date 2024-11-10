import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F 
from dezero.models import MLP

x = Variable(np.array([[1,2,3],
                       [4,5,6]]))

indices = np.array([0,0,1])
y = F.get_item(x, indices)
print(y)

y = x[1]
print(y)
y = x[:2]
print(y)

model = MLP((10, 3))
x = np.array([[0.2, -0.4]])
y = model(x)
print(y)

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

p = softmax1d(y)
print(p)


model = MLP((10, 3))
x = np.array([[0.2, -0.4],
              [0.3, 0.5],
              [1.3, -3.2],
              [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)
