import os
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
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class test_dataset:
    def __init__(self, image_root, gt_root):
        self.images = [image_root + '/' + f for f in os.listdir(image_root)]
        self.gts = [gt_root + '/' + f for f in os.listdir(gt_root)]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.ToTensor()
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


def eval(args):

    model = 'DarkraNet'
    Datasets = ['CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', 'CVC-300', 'NBI']

    for dataset in Datasets:

        print('-----> Processing {} dataset ...'.format(dataset))

        mIoU = 0.
        mDice = 0.
        F2 = 0.
        precision = 0.
        recall = 0.
        gt_root = os.path.join(args.data_path, dataset, 'masks')
        image_root = os.path.join(args.result_path, model, dataset)
        test_loader = test_dataset(image_root, gt_root)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image
            data = image[0, 1, :, :]
            data = np.array(data)
            target = np.array(gt)
            N = gt.shape

            input_flat = np.reshape(data, (-1))
            target_flat = np.reshape(target, (-1))

            mIoU += cal(input_flat, target_flat)[0]
            mDice += cal(input_flat, target_flat)[1]
            F2 += cal(input_flat, target_flat)[2]
            precision += cal(input_flat, target_flat)[3]
            recall += cal(input_flat, target_flat)[4]

        mIoU /= test_loader.size
        mDice /= test_loader.size
        F2 /= test_loader.size
        precision /= test_loader.size
        recall /= test_loader.size

        print('\nDataset: {}'.format(dataset))
        print('mIoU: {:.4f} mDice: {:.4f} F2: {:.4f} Precision {:.4f} Recall {:.4f}\n'
                 .format(mIoU, mDice, F2, precision, recall))

def cal(predictions, ground_truth):

    SMOOTH = 1e-6
    tp = np.sum(predictions * ground_truth)
    tn = np.sum(predictions * ground_truth) - tp
    fp = np.sum((1 - ground_truth) * predictions)
    fn = np.sum((1 - predictions) * ground_truth)

    iou = tp/(tp+fp+fn+SMOOTH)
    dice = (2*tp)/(2*tp+fp+fn+SMOOTH)
    precision = tp/(tp+fp+SMOOTH)
    recall = tp/(tp+fn+SMOOTH)
    f2 = (5*precision*recall)/(4*precision+recall+SMOOTH)

    return iou, dice, f2, precision, recall




def parse_args():
    parser = argparse.ArgumentParser(description='Polyp segmentation evaluation')
    parser.add_argument('--result_path', default='./results', type=str, help='Result map path')
    parser.add_argument('--data_path', default='./TestDataset', type=str, help='Test data directory')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    eval(args)
