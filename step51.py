# %%
from dezero import optimizers
from dezero.dataloaders import DataLoader
import dezero.datasets
import numpy as np
import matplotlib.pyplot as plt
from dezero.models import MLP
import dezero.functions as F
# %%
# def f(x: np.ndarray):
#     x = x.flatten()
#     x = x.astype(np.float32)
#     x /= 255.0
#     return x

# train_set = dezero.datasets.MNIST(train=True, transform=f)
# test_set = dezero.datasets.MNIST(train=False, transform=f)

# print(len(train_set))
# print(len(test_set))
# # %%
# x, t = train_set[0]
# print(type(x), x.shape)
# print(t)
# # %%
# x, t = train_set[0]
# plt.imshow(x.reshape(28, 28), cmap='gray')

# %%
#
max_epoch = 54
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size)

model = MLP((hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch + 1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(train_set),
        sum_acc / len(train_set)
    ))

    sum_loss, sum_acc = 0, 0
    for x, t in test_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss/len(test_set),
        sum_acc/len(test_set)
    ))
