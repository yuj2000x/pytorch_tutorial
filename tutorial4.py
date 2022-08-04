# OPTIMIZING MODEL PARAMETERS

import torchvision.models as models
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# Download data
training_data = datasets.FashionMNIST(
    root="data_fashion",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data_fashion",
    train=False,
    download=False,
    transform=ToTensor()
)

# 使用 dataloader load 資料
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)  # 一個 batch 裡有 64 筆資料


# Define nn
class NeuralNetwork(nn.Module):
    def __init__(self):  # 建構子
        # super 會建立 bounded 的 super object
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(  # 快速搭建線性 ReLU 神經網路
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):  # 方法
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 因為有 self 所以要具體化這個 class
model = NeuralNetwork()

# Define hyperparameter
learning_rate = 1e-1
batch_size = 64
epochs = 30


# 定義訓練模型
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 有 enumerate 迴圈就可以不用寫 ++i 之類的 他自己會列舉 0 1 2 3...
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # model input data X
        pred = model(X)
        # result of pred 和 correct answer y 做 loss(cross entropy)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # 將梯度至零
        loss.backward()        # 反向傳播
        optimizer.step()       # 優化

        if batch % 100 == 0:   # 每 100 個 batch 顯示一次
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 定義測試結果
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():  # ?_?
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()  # 一個 batch 生出一個 loss
            correct += (pred.argmax(1) ==
                        y).type(torch.float).sum().item()  # ?_?

    test_loss /= num_batches  # total loss 除以 batch總數(共157個)
    correct /= size
    print(correct)
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 選擇 define loss function 好壞的算法
loss_fn = nn.CrossEntropyLoss()
# 選能快速找到最好 loss function 的方法
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
