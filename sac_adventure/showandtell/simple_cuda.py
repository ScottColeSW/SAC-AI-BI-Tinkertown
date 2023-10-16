import torch
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)

print(use_cuda)
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Specify the GPU device index to use (e.g., '0' for the first GPU)
import tensorflow as tf
print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")
print()


import GPUtil
GPUtil.getAvailable()