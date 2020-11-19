import torch
import numpy as np
from PIL import Image, ImageFilter
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision.datasets import ImageNet
from torchvision.transforms import RandomCrop, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, Compose, ToPILImage, CenterCrop, Normalize, Grayscale, Resize
from tqdm import tqdm

class Dataset(data.Dataset):
    def __init__(self, settings, validation=None, mode=None):
      #  dataset = settings['general']['dataset']
        dataset = settings
        self.size = 0
        if dataset == 'imageNET':
            self.images = 'data/imageNET' 
            #self.labels = pathtolabels
            self.size = 1
                
        elif dataset == 'YFCC100M':
            self.images = 'data/YFCC100M'
           # self.labels = pathtolabels    
            #self.size = len(self.images)
            
    def __getitem__(self, index):
        idx = np.random.randint(0,self.size)
        img_batch = np.load(f'{self.images}/train_data_batch_1', allow_pickle = True)
        return img_batch['data'][idx]
        #return ImageNet(self.images).loader('train_data_batch_1')
        
    def __len__(self):
        return self.size
    
def load_data(settings, transformation=None, n_train=None, n_test=None):
    ds = Dataset(settings)
    try:
        dataloader = torch.utils.data.DataLoader(
            ds,
            shuffle=True,
            batch_size = 1
        )
        dl = dataloader
        for items in tqdm(dl):
            plt.imshow(items)
            
    except UnboundLocalError:
        print('error')
    
   
if __name__ == "__main__":
    
    load_data('imageNET')
