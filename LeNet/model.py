from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        self.s2 = nn.AvgPool2d(2, 2)
        self.c3 = nn.Conv2d(6, 16, 5)
        self.s4 = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(400, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x
