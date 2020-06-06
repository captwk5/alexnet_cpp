#%%
import os
import sys
import glob
import cv2

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from pytorch_network import *
from get_init_weight import *

from torch.hub import load_state_dict_from_url

CUDA_FLAG = False

def train():

    device = torch.device("cuda:0")

    dataset = torchvision.datas2ets.ImageFolder(root="data/Dabeeo2F224/",
    # dataset = torchvision.datasets.ImageFolder(root="data/test224/",
                            transform=transforms.Compose([
                                transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                                #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #     std=[0.229, 0.224, 0.225])
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
                            ]))

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=0)


    testset = torchvision.datasets.ImageFolder(root="data/Dabeeo2F224Test_Origin/",
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
                            ]))
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0)

    # net = AlexNet()
    net = AlexNetTorch()
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', progress=True)
    net.load_state_dict(state_dict)

    # save_weight_json(net)

    # net = set_weight(net)

    # net = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    # net = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=True)
    # net = VGG16()
    net.to(device)
    print(net)
    # print(net.state_dict()['fc3.weight'][0])
    #net.state_dict()['fc3.weight'][0][0] = 1
    #print(net.state_dict()['fc3.weight'][0])

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(pytorch_total_params)

    EPOCH = 3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    data_size = len(dataloader)
    print("Total Data : {}".format(data_size))

    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i == data_size - 1:
                print('Epoch : %d, loss: %.10f' % (epoch + 1, (running_loss / data_size)))
                # if (running_loss / data_size) < 0.0001:
                    # stop_flag = True
                running_loss = 0.0

    # PATH = './Dabeeo2F-' + str(epoch) + '.pth'
    # torch.save(net.state_dict(), PATH)
    # net.load_state_dict(torch.load('./Dabeeo2F-2.pth'))
    # net.eval()
    for data in testloader:
        test_images, labels = data
        test_images = test_images.to(device)
        outputs = net(test_images)
        m = nn.Softmax(dim = 1)
        output_prob = m(outputs)
        _, predicted = torch.max(outputs.data, 1)
        print(str(dataset.classes[int(labels[0])]) + " -> " + str(dataset.classes[int(predicted[0])]) + " : " + str(torch.max(output_prob, 1).values))

    net.cpu()
    net.eval()
    example = torch.rand(1, 3, 224, 224)
    # net = torch.quantization.convert(net)
    traced_script_module = torch.jit.trace(net, example)
    traced_script_module.save('./Dabeeo2F_AUG.pt')

    # net.to(device)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

if __name__ == "__main__":
    CUDA_FLAG = torch.cuda.is_available()

    main()
