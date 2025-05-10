import os
import copy
import time
import torch
import pandas as pd
from torch import nn
from model import LeNet
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# 训练集数据
def data_process():
    train_data = FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    train_data, val_data = data.random_split(train_data, [round(len(train_data)*0.8), round(len(train_data)*0.2)])
    train_loader = data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(dataset=val_data, batch_size=64, num_workers=0)
    return train_loader, val_loader

# 模型训练
def train(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 设定优化器和损失函数
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam优化器(梯度下降)

    # 开始训练
    train_losses = list()
    val_losses = list()
    train_accs = list()
    val_accs = list()
    best_acc = float(0)
    best_model = None
    start = time.time()
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs) + '\n' + "-" * 20)
        train_loss, train_acc, val_loss, val_acc = float(0), float(0), float(0), float(0)
        train_num, val_num = int(0), int(0)
        # 前向传播
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.train()
            outputs = model(inputs)
            pre_labels = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(pre_labels == labels)
            train_num += inputs.size(0)
        # 反向传播
        for step, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.eval()
            outputs = model(inputs)
            pre_labels = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_acc += torch.sum(pre_labels == labels)
            val_num += inputs.size(0)
        # 计算并显示损失值和正确率
        train_losses.append(train_loss / train_num)
        train_accs.append(train_acc.item() / train_num)
        val_losses.append(val_loss / val_num)
        val_accs.append(val_acc.item() / val_num)
        print("{} train loss:{:.4f} train acc:{:.4f}".format(epoch+1, train_losses[-1], train_accs[-1]))
        print("{} val loss:{:.4f} val acc:{:.4f}".format(epoch+1, val_losses[-1], val_accs[-1]))
        # 记录最佳模型参数
        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            best_model = copy.deepcopy(model.state_dict())
        # 显示单轮训练时间
        end = time.time() - start
        print("Training complete in {:.0f}min {:.2f}s".format(end//60, end%60) + '\n')
        start = time.time()

    # 保存模型最佳参数
    if not os.path.exists('./LeNet'): os.makedirs('./LeNet')
    torch.save(best_model, "./LeNet/best_model.pth")

    # 整合训练过程参数
    train_process = pd.DataFrame(data={
        "epoch": range(epochs),
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
    })
    return train_process

# 损失值及正确率可视化
def matplot_loss(train_process):
    plt.figure(figsize=(12, 4))
    # 损失值折线图
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process["train_loss"], 'ro-', label='train loss')
    plt.plot(train_process["epoch"], train_process["val_loss"], 'bs-', label='val loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # 正确率折线图
    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"], train_process["train_acc"], 'ro-', label='train acc')
    plt.plot(train_process["epoch"], train_process["val_acc"], 'bs-', label='val acc')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

if __name__ == '__main__':
    LeNet = LeNet()
    train_data, val_data = data_process()
    train_process = train(LeNet, train_data, val_data, epochs=20)
    matplot_loss(train_process)

