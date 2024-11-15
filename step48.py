from matplotlib import pyplot as plt
import dezero.datasets
import math
import numpy as np 
from dezero import optimizers
import dezero.functions as F 
from dezero.models import MLP


train_set = dezero.datasets.Spiral2()


# 하이퍼파라미터 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# 데이터 읽기/모델, 옵티마이저 생성
# x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

loss_list = []
for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i+1)* batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        # 기울기 산출/매개변수 갱신
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * (len(batch_t))


    # 에포크마다 학습 경과 출력
    avg_loss = sum_loss /data_size
    print(f"epoch {epoch + 1}, loss: {avg_loss}")

    loss_list.append(avg_loss)



# Plot the loss curve
plt.plot(range(max_epoch), loss_list)
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.show()


