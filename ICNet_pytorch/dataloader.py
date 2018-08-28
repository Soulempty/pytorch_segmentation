import numpy as np
import torch
from PIL import Image
import random
import os
from img_proc import *
import scipy.misc as m
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Normalize,Resize
from torch.utils.data import Dataset

#num_classes=2
#full_to_train={}
class RandomFlip(object):   
    def __call__(self, img, label):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT) #left or right
            return img, label
        return img, label

class RandomResize(object):
    def __init__(self):
        self.p = random.choice([0.5,0.75,1.0,1.5,1.75,2])
    def __call__(self, img, label):
        w=img.size[0]*self.p
        h=img.size[1]*self.p
        img = img.resize((w,h),Image.BILINEAR)
        label = label.resize((w,h),Image.NEAREST) #left or right
        return img, label

class MyTransform(Dataset):
    mean_rgb = {
        "bdd": [103.939, 116.779, 123.68],
        "city": [73.936, 74.613, 71.064],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self,data_txt,augmentations=None,is_transform=True): 
        
        self.mean = np.array(self.mean_rgb["city"])
        self.augmentations=augmentations
        self.is_transform=is_transform
        self.img_norm=True
        self.inputs = []
        self.targets = []
        path="/home1/"
        filenames=open(data_txt).readlines()
        FLAG=True
        for file in filenames:
            self.inputs.append(path+file.split()[0])
            self.targets.append(path+file.split()[1])
            if FLAG:
                print("img_path:",path+file.split()[0])
                print("label_path:",path+file.split()[1])
                FLAG=False
    def __len__(self):
        return len(self.inputs)  
    def __getitem__(self, i):
        # do something to both images and labels
        input, target = Image.open(self.inputs[i]), Image.open(self.targets[i])
        #input,target=np.array(input, dtype=np.uint8),np.array(target, dtype=np.uint8)
        if self.augmentations is not None:
            img, label = self.augmentations(input,target)
            
        if self.is_transform:
            img, label1,label4,label24 = self.transform(img, label)
        return img, label1,label4,label24
    def transform(self, img, label):
        #img = img.resize(self.resize,Image.BILINEAR)
        #label1=label.resize(self.resize,Image.NEAREST).astype(int)
        assert img.size == label.size and img.size==(640,360)
        label1=np.array(label).astype(int)
        label4=np.array(label.resize((40,23),Image.NEAREST)).astype(int)
        label24=np.array(label.resize((80,45),Image.NEAREST)).astype(int)
        # uint8 with RGB mode
        img=np.array(img)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        label1 = torch.from_numpy(label1).long()
        label4 = torch.from_numpy(label4).long()
        label24 = torch.from_numpy(label24).long()
        return img,label1,label4,label24


