from PIL import Image
import argparse
import torch
import os
from torchvision import transforms

model = torch.load("net.pkl")["model"]


argparser = argparse.ArgumentParser()
argparser.add_argument("-f")

args = argparser.parse_args()

if os.path.exists(args.f):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(args.f)
    tensor = torch.unsqueeze(transform(image), dim=0)
    p = model.predict(tensor.cuda())
    print(p)
