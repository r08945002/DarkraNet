import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from sklearn.utils import shuffle
import pandas as pd



class PolypDataset(Dataset):
    def __init__(self, data_path, csv, mode):
        """
        dataloader for polyp segmentation
        """
        self.images = None
        self.masks = None
        self.data = data_path
        self.csv = shuffle(pd.read_csv(csv))
        self.mode = mode

    def preprocess_img(self, images):

        if self.mode == 'Test':
            trans = transforms.Compose([transforms.Resize((352, 352)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                       ])


        else:
            trans = transforms.Compose([transforms.Resize((352, 352)),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        # transforms.ColorJitter(brightness=0.5),
                                        transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                       ])

        return trans(images)

    def preprocess_mask(self, masks):

        if self.mode == 'Test':
            trans = transforms.Compose([transforms.Resize((352, 352)),
                                        transforms.ToTensor(),
                                       ])

        else:
            trans = transforms.Compose([transforms.Resize((352, 352)),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        # transforms.ColorJitter(brightness=0.5),
                                        transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                                        transforms.ToTensor()
                                       ])

        return trans(masks)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_id = self.csv.loc[index, "Image_id"]
        data_dir = self.csv.loc[index, "Data_dir"]
        image_path = os.path.join(self.data, data_dir, 'images', image_id)
        mask_path = os.path.join(self.data, data_dir, 'masks', image_id)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        seed = np.random.randint(2021) # make a seed with numpy generator
        random.seed(seed) # apply this seed to img transforms
        torch.manual_seed(seed)
        image = self.preprocess_img(image)

        random.seed(seed) # apply this seed to make img transforms = mask transforms
        torch.manual_seed(seed)
        mask = self.preprocess_mask(mask)

        return image, mask

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.csv)




if __name__=='__main__':

    data_path = './Polyp_dataset'
    trainset = PolypDataset(data_path=data_path, csv='train.csv', mode='Train')
    valset = PolypDataset(data_path=data_path, csv='val.csv', mode='Train')

    print('# images in trainset:', len(trainset))
    print('# images in valset:', len(valset))
