import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import argparse
import torch.nn.functional as F
import warnings
from dataloader import PolypDataset
from seg_model import DarkraNet
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

fix random seeds for reproducibility
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
    parser.add_argument('--result_path', default='./results', type=str, help='Result map path')
    parser.add_argument('--data_path', default='./TestDataset', type=str, help='Test data directory')
    return parser.parse_args()

def predict(args):

    Datasets = ['CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', 'CVC-300', 'NBI']

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)


    model = DarkraNet()
    model.load_state_dict(torch.load('DarkraNet_final.pth')) # put your best model path

    model.to(DEVICE)
    model.eval()
    print('-----> DarkraNet start predicting ...\n')

    for dataset in Datasets:

        print('-----> Processing {} dataset ...'.format(dataset))
        gt_root = os.path.join(args.data_path, dataset, 'masks')
        image_root = os.path.join(args.data_path, dataset, 'images')
        test_loader = test_dataset(image_root, gt_root, args.testsize)
        save_path = os.path.join(args.result_path, 'DarkraNet', dataset)
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
            thresh = threshold_otsu(res)
            binary = res > thresh
            plt.imsave(os.path.join(save_path, name), binary, cmap=plt.cm.gray)



    print('\nDone !')

if __name__=='__main__':

    args = parse_args()

    predict(args)

