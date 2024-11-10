from matplotlib import pyplot as plt
import numpy as np
from dezero.datasets import Spiral
from dezero import (
    DataLoader,
    no_grad
)
from dezero.models import MLP
from dezero.optimizers import SGD
import dezero.functions as F


batch_size = 10
max_epoch = 1

train_set = Spiral(train=True)
test_set = Spiral(train=True)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape)
        break

    for x, t in test_loader:
        print(x.shape, t.shape)
        break



y = np.array([[0.2, 0.8, 0],
              [0.1, 0.9, 0],
              [0.8, 0.1, 0.1]])

t = np.array([1, 2, 0])
acc = F.accuracy(y, t)
print(acc)

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

model = MLP((hidden_size, 3))
optimizer = SGD(lr).setup(model)

train_avg_loss = []
train_avg_acc = []
test_avg_loss = []
test_avg_acc = []

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

    print(f"epoch: {epoch+1}")
    avg_loss = sum_loss/len(train_set)
    avg_acc = sum_acc/len(train_set)
    print(f"train loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")

    train_avg_loss.append(avg_loss)
    train_avg_acc.append(avg_acc)

    sum_loss, sum_acc = 0, 0
    with no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    avg_loss = sum_loss/len(train_set)
    avg_acc = sum_acc/len(train_set)
    test_avg_loss.append(avg_loss)
    test_avg_acc.append(avg_acc)

    print(f"test loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")


plt.figure(figsize=(8, 12))

# Loss plot
plt.subplot(2, 1, 1)
plt.plot(range(max_epoch), train_avg_loss, label='train')
plt.plot(range(max_epoch), test_avg_loss, label='test') 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss during training')
plt.legend()

# Accuracy plot
plt.subplot(2, 1, 2)
plt.plot(range(max_epoch), train_avg_acc, label='train')
plt.plot(range(max_epoch), test_avg_acc, label='test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy during training')
plt.legend()

plt.tight_layout()
plt.show()


        
