import os
import cv2
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import argparse
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from dataloader import PolypDataset
from seg_model import DarkraNet
from other_models import PraNet, HarDMSEG
from video import vid2frame, frame2vid


# fix random seeds for reproducibility
SEED = 7414
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + '/' + f for f in os.listdir(image_root)]
        self.gts = [gt_root + '/' + f for f in os.listdir(gt_root)]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        if self.testsize == 'None':
            image = cv2.imread(self.images[self.index])
            gt = cv2.imread(self.gts[self.index])
        else:
            image = Image.open(self.images[self.index]).convert('RGB')
            image = self.transform(image).unsqueeze(0)
            gt = Image.open(self.gts[self.index]).convert('L')
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, gt, name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='Testing size')
    parser.add_argument('--result_path', default='./output_frame', type=str, help='Result map path')
    parser.add_argument('--video_name', default='WL.mp4', type=str, help='Colonoscopy video name')

    return parser.parse_args()

def video_seg(args):

    print('-----> Processing {} ...\n'.format(args.video_name))
    fps, frame_path = vid2frame(args.video_name)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)


    model = DarkraNet()
    model.load_state_dict(torch.load('DarkraNet.pth')) # put your best model path

    model.to(DEVICE)
    model.eval()
    print('-----> Start predicting ...\n')

    gt_root = frame_path
    image_root = frame_path
    test_loader = test_dataset(image_root, gt_root, args.testsize)
    save_path = os.path.join(args.result_path, frame_path)
    os.makedirs(save_path, exist_ok=True)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(DEVICE)

        res5, res3, res2 = model(image)
        res = res5 + res2
        res = torch.div(res, 2)

        res = F.upsample(res, size=gt.shape, mode='bicubic', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        plt.imsave(os.path.join(save_path, name), res, cmap=plt.cm.gray)


    gt_root = save_path
    test_loader = test_dataset(image_root, gt_root, 'None')
    for i in range(test_loader.size):

        image, gt, name = test_loader.load_data()
        thresh = cv2.Canny(gt, 128, 256)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            if (hierarchy[0, i, 3] == -1):
                cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(save_path, name), image)


    print('-----> Processing the output video ...\n')
    frame2vid(save_path, args.video_name, fps)

if __name__=='__main__':

    args = parse_args()

    video_seg(args)
