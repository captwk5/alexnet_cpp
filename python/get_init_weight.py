import os
import sys

import csv
import json
import struct

import numpy as np
from tqdm import tqdm

def save_weight_json(net):
    if os.path.isdir("alexnet_weight_init") is False:
        os.mkdir("alexnet_weight_init")
    # else:
    #     os.remove("alexnet_weight_init/*.bin")

    bias = 0

    for i, n in tqdm(enumerate(net.parameters(), 0)):
        cpu_tensor = n.cpu()
        tensor = cpu_tensor.detach().numpy()

        if len(tensor.shape) is 4:
            f = open('alexnet_weight_init/' + str(tensor.shape[0]) + "-" + str(tensor.shape[1]) + "-"
                    + str(tensor.shape[2]) + "-" + str(tensor.shape[3]) + '.bin', 'wb')
            for i, t0 in enumerate(tensor, 0):
                for j, t1 in enumerate(t0, 0):
                    for k, t2 in enumerate(t1, 0):
                        for p, t3 in enumerate(t2, 0):
                            b = struct.pack('f',np.float(t3))
                            f.write(b)
            f.close()
        elif len(tensor.shape) is 2:
            f = open('alexnet_weight_init/' + str(tensor.shape[0]) + "-" + str(tensor.shape[1]) + '.bin', 'wb')
            for i, t0 in enumerate(tensor, 0):
                for j, t1 in enumerate(t0, 0):
                    b = struct.pack('f',np.float(t1))
                    f.write(b)
            f.close()
        else:
            f = open('alexnet_weight_init/' + str(tensor.shape[0]) + '-' + str(bias) + 'b.bin', 'wb')
            for i, t0 in enumerate(tensor, 0):
                b = struct.pack('f',np.float(t0))
                f.write(b)
            f.close()
            bias += 1
    