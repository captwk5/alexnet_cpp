import os
import sys
import cv2
import glob
import torch
import time
import struct

import threading

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import tensorflow as tf

from pytorch_network import *
from tqdm import tqdm
import cython
import numpy as np

from multiprocessing.pool import ThreadPool

dataset = torchvision.datasets.ImageFolder(root="data/Dabeeo2F224_Origin/",
# dataset = torchvision.datasets.ImageFolder(root="data/IndoorCVPR22409/",
                           transform=transforms.Compose([
                               transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                            #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #     std=[0.229, 0.224, 0.225])
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
                           ]))

testset = torchvision.datasets.ImageFolder(root="data/Dabeeo2F224Test_Origin/",
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                            #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #     std=[0.229, 0.224, 0.225])         # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
                           ]))
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0)

print(dataset.classes)

net = AlexNetTorch()

def set_weight(net):
    f_size = struct.calcsize('f')

    idx = 0

    for n in tqdm(net.state_dict().keys()):
        if len(net.state_dict()[n].shape) == 4:
            l1, l2, l3, l4 = (net.state_dict()[n].shape)
            f = open("alexnet_weight_init/" + str(l1) + "-" + str(l2) + "-" + str(l3) + "-" + str(l4) + ".bin", 'rb')
            # f = open("alexnet_weight_train/" + str(l1) + "-" + str(l2) + "-" + str(l3) + "-" + str(l4) + "-c.bin", 'rb')

            data = f.read(f_size * l1 * l2 * l3 * l4)
            data_list = struct.unpack('f' * l1 * l2 * l3 * l4, data)
            data_list = list(data_list)
            data_list = torch.FloatTensor(data_list)
            data_list = torch.reshape(data_list, (l1, l2, l3, l4))

            data_list = torch.nn.Parameter(data_list)

            if n == 'features.0.weight':
                net.features[0].weight = data_list
            elif n == 'features.3.weight':
                net.features[3].weight = data_list
            elif n == 'features.6.weight':
                net.features[6].weight = data_list
            elif n == 'features.8.weight':
                net.features[8].weight = data_list
            elif n == 'features.10.weight':
                net.features[10].weight = data_list

            f.close()
        elif len(net.state_dict()[n].shape) == 2:
            l1, l2 = (net.state_dict()[n].shape)
            f = open("alexnet_weight_init/" + str(l1) + "-" + str(l2) + ".bin", 'rb')
            # f = open("alexnet_weight_train/" + str(l1) + "-" + str(l2) + "-c.bin", 'rb')

            data = f.read(f_size * l1 * l2)
            data_list = struct.unpack('f' * l1 * l2, data)

            data_list = list(data_list)
            data_list = torch.FloatTensor(data_list)
            data_list = torch.reshape(data_list, (l1, l2))

            data_list = torch.nn.Parameter(data_list)

            if n == 'classifier.1.weight':
                net.classifier[1].weight = data_list
            elif n == 'classifier.4.weight':
                net.classifier[4].weight = data_list
            elif n == 'classifier.6.weight':
                net.classifier[6].weight = data_list

            f.close()
        elif len(net.state_dict()[n].shape) == 1:
            l1 = (net.state_dict()[n].shape[0])

            f = open("alexnet_weight_init/" + str(l1) + "-" + str(idx) + "b.bin", 'rb')
            # f = open("alexnet_weight_train/" + str(l1) + "-" + str(idx) + "b-c.bin", 'rb')

            data = f.read(f_size * l1)
            data_list = struct.unpack('f' * l1, data)

            data_list = list(data_list)
            data_list = torch.FloatTensor(data_list)

            data_list = torch.nn.Parameter(data_list)

            if n == 'features.0.bias':
                net.features[0].bias = data_list
            elif n == 'features.3.bias':
                net.features[3].bias = data_list
            elif n == 'features.6.bias':
                net.features[6].bias = data_list
            elif n == 'features.8.bias':
                net.features[8].bias = data_list
            elif n == 'features.10.bias':
                net.features[10].bias = data_list
            elif n == 'classifier.1.bias':
                net.classifier[1].bias = data_list
            elif n == 'classifier.4.bias':
                net.classifier[4].bias = data_list
            elif n == 'classifier.6.bias':
                net.classifier[6].bias = data_list

            f.close()
            idx += 1
    
    return net

net = set_weight(net)

device = torch.device("cuda:0")
net.to(device)

for data in testloader:
    test_images, labels = data
    test_images = test_images.to(device)
    outputs = net(test_images)
    m = nn.Softmax(dim = 1)
    output_prob = m(outputs)
    _, predicted = torch.max(outputs.data, 1)
    if int(predicted[0]) < 39:
        print(str(dataset.classes[int(labels[0])]) + " -> " + str(dataset.classes[int(predicted[0])]) + " : " + str(torch.max(output_prob, 1).values))
    else:
        print("Predict Fail!!")

# net.cpu()
# net.eval()
# example = torch.rand(1, 3, 224, 224)
# # net = torch.quantization.convert(net)
# traced_script_module = torch.jit.trace(net, example)
# traced_script_module.save('./Dabeeo2F-fltest.pt')

# from data.data_processing.image_augmentation import *

# im_cls_list = glob.glob("/media/4tb/IPS/datas/office_20200428/extract_img/test_data/*")

# for i in im_cls_list:
#     im_list = glob.glob(i + "/*.png")
#     c = i.split('/')
#     c_str = c[len(c) - 1]

#     data_path = "data/DabeeoOffice11FTest224/" + c_str + "/"

#     if os.path.isdir(data_path) is False:
#         os.mkdir(data_path)

#     for img in im_list:
#         img_name = img.split("/")
#         name = img_name[len(img_name) - 1]
#         img = cv2.imread(img)
#         img = image_centercrop_resize(img)
#         cv2.imwrite(data_path + "/" + name, img)
