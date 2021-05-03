import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import AlexNet

if os.path.exists("net.pkl"):
    pkl = torch.load("net.pkl")
    net = pkl.get("model")
    sepoch = pkl.get("epoch")
else:
    net = AlexNet().cuda()
    sepoch = 1

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(params=net.parameters(), lr=1e-2, momentum=9e-1)

data_loader = DataLoader(dataset=ImageFolder("data/train",
                                             transform=transforms.Compose([
                                                 transforms.Resize((256, 256)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                             ])),
                         batch_size=50,
                         shuffle=True)


def adjust_learning_rate(epoch):
    lr = 1e-2 * 1e-1 ** (epoch // 20)
    for group in optimizer.param_groups:
        group['lr'] = lr


def train():
    net.train()
    for epoch in range(sepoch, 101):
        losses = 0
        batch_count = 0

        adjust_learning_rate(epoch=epoch)

        for batch, (samples, targets) in enumerate(data_loader, start=1):
            output = net(samples.cuda())
            optimizer.zero_grad()

            loss = criterion(output, targets.cuda())
            loss.backward()
            optimizer.step()

            losses += loss.item()
            batch_count += 1
            print(
                "Batch {batch} Loss: {loss:.4f}".format(
                    batch=batch,
                    loss=loss.item()
                )
            )
        print(
            "Epoch {epoch} Loss: {losses:.4f}".format(
                epoch=epoch,
                losses=losses / batch_count
            )
        )
        torch.save({"model": net, "epoch": epoch + 1}, "net.pkl")


if __name__ == "__main__":
    train()
