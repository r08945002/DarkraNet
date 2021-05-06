import time
import torch
import torch.nn as nn
import random
import numpy as np
from seg_model import DarkraNet

# fix random seeds for reproducibility
SEED = 1314
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def speed(model, name):
    model.eval()
    image = torch.rand(1, 3, 352, 352).cuda()

    inference_times = []
    for i in range(0, 30):
        start_time = time.time()
        with torch.no_grad():
            output = model(image)

            torch.cuda.synchronize()

        end_time = time.time()
        inference_times.append(end_time - start_time)

    inference_time = sum(inference_times) / len(inference_times)
    fps = 1.0 / inference_time
    print('%10s : %fs %10s : %f' % (name, inference_time, 'fps', fps))


if __name__ == '__main__':

    DarkraNet = DarkraNet().cuda()

    speed(DarkraNet, 'DarkraNet')

