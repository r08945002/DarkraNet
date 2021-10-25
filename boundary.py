import os
import cv2
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import argparse
from scipy import misc
import torch.nn.functional as F
import warnings
from dataloader import PolypDataset
from seg_model import DarkraNet

# fix random seeds for reproducibility
SEED = 1122
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
        image = cv2.imread(self.images[self.index])
        gt = cv2.imread(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, gt, name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='Testing size')
    parser.add_argument('--result_path', default='./boundary', type=str, help='Result map path')
    parser.add_argument('--mask_path', default='./results', type=str, help='Result map path')
    parser.add_argument('--data_path', default='./TestDataset', type=str, help='Test data directory')
    return parser.parse_args()

def predict(args):

    Datasets = ['CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', 'CVC-300', 'NBI']

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)


    model = DarkraNet()
    model.load_state_dict(torch.load('DarkraNet_best.pth')) # put your best model path

    model.to(DEVICE)
    model.eval()
    print('-----> DarkraNet start predicting ...\n')

    for dataset in Datasets:

        print('-----> Processing {} dataset ...'.format(dataset))
        gt_root = os.path.join(args.mask_path, 'DarkraNet', dataset)
        image_root = os.path.join(args.data_path, dataset, 'images')
        test_loader = test_dataset(image_root, gt_root, args.testsize)
        save_path = os.path.join(args.result_path, 'DarkraNet', dataset)
        os.makedirs(save_path, exist_ok=True)
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()

            thresh = cv2.Canny(gt, 128, 256)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i, cnt in enumerate(contours):
                if (hierarchy[0, i, 3] == -1):
                    cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)

            cv2.imwrite(os.path.join(save_path, name), image)

    print('\nDone !')



if __name__=='__main__':

    args = parse_args()

    predict(args)

