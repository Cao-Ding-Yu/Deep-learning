import torch
from model import LeNet
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# 测试集数据
def data_process():
    test_data = FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    loader = data.DataLoader(dataset=test_data, batch_size=64, num_workers=0)
    return loader

# 模型评估测试
def test_model(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    test_correct, test_num = float(0), int(0)
    with torch.no_grad():
        for img, label in data_loader:
            img = img.to(device)
            label = label.to(device)
            pred = torch.argmax(model(img), dim=1)
            test_correct += torch.sum(pred == label)
            test_num += img.size(0)
    test_correct = test_correct.item() / test_num
    print('Accuracy of the network on the test images: %.2f %%' % (test_correct * 100))

if __name__ == '__main__':
    LeNet = LeNet()
    LeNet.load_state_dict(torch.load("./LeNet/best_model.pth"))
    test_model(LeNet, data_process())
