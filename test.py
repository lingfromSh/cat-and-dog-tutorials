import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import AlexNet

net = AlexNet().cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(params=net.parameters(), lr=1e-2, momentum=9e-1)

data_loader = DataLoader(dataset=ImageFolder("data/test",
                                             transform=transforms.Compose([
                                                 transforms.Resize((256, 256)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                             ])),
                         batch_size=50,
                         shuffle=True)

pkl = torch.load("net.pkl")
net = pkl.get("model")
epoch = pkl.get("epoch")

net.eval()

correct = 0
total = 0
for samples, targets in data_loader:
    output = net(samples.cuda())

    top = output.topk(1, 1, True, True).indices

    for p,r in zip(top, targets.cuda()):
        if p.eq(r):
            correct += 1
        total += 1
    
print(f"Acc: {correct / total}")