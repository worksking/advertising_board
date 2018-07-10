import numpy as np
import os
from os.path import join as join
from os.path import abspath
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import random

class Dataset(data.Dataset):
    '''

    '''
    def __init__(self, root, data_dir='train', transform=transforms.ToTensor(), loader=None):
        assert data_dir in ('train', 'val', 'test', 'test2'), "data_dir must be 'train', 'val' or 'test'"
        self.root = root
        self.transforms = transform
        self.loader = loader
        self.images = []
        self.names = []
        self.labels = []
        self.dir = [] 
        self.files = []
        self.data_dir = data_dir
    
        if self.data_dir == 'train':
            with open (join(self.root, 'train.txt'), 'r') as f:
                filemap = [x.strip() for x in f.readlines()]
                filemap = [x.split(" ") for x in filemap]
                filemap = dict(filemap)
            for dir, _, files in os.walk(join(self.root, self.data_dir)):
                self.images += [join(self.root, self.data_dir, file) for file in files]
                self.labels += [(int(filemap[file])-1) for file in files]
            # print(' the length imgs and labels:', len(self.images), len(self.labels))
            # for i in range(len(self.images)):
            #     print(self.images[i], self.labels[i])

        elif self.data_dir == 'test':
            with open(join(self.root, 'test.txt'), 'r') as f:
                files = [x.strip() for x in f.readlines()]
            for file in files:
                self.images += [join(self.root, self.data_dir, file)]
                self.files += [file]
            print(' the length imgs and files:', len(self.images), len(self.files)) 
            # for i in range(len(self.images)):
            #     print(self.images[i], self.files[i])        

            

    def _loader(self, path):
        return Image.open(path).convert('L').resize((128,128)).rotate(random.randint(-80,80))
        # return Image.open(path).convert('RGB').resize((64,64))

    def image_loader(self,path):
        return Image.open(path).convert('RGB').resize((64, 64))

    def __getitem__(self, index):
        if self.loader is None:
            self.loader = self._loader
        
        if self.data_dir =='test':
            self.loader = self.image_loader
            imgs = self.images[index]
            imgs = self.transforms(self.loader(imgs))
            files = self.files[index]
            return imgs, files
            # return imgs

        imgs = self.images[index]        
        imgs = self.transforms(self.loader(imgs))
        lables = self.labels[index]
        return imgs, lables
    
    def __len__(self):
        return len(self.images)


# '''
# test file
# '''
# def main():
#     test_datasets = Dataset("/data1/Adam/advertising_board/datasets/", 'train')

# if __name__ == '__main__':
#     main()
    
