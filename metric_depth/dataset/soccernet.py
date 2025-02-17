import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class SoccerNet(Dataset):
    def __init__(self, images_path, mode, size=(294,294)):
        self.mode = mode
        self.size = size
        self.images_path,self.depth_path = self.images_list_generator(self,images_path)

        net_w, net_h = size
        self.transform = Compose([
                                     Resize(
                                         width=net_w,
                                         height=net_h,
                                         resize_target=True if mode == 'train' else False,
                                         keep_aspect_ratio=True,
                                         ensure_multiple_of=14,
                                         resize_method='lower_bound',
                                         image_interpolation_method=cv2.INTER_CUBIC,
                                     ),
                                     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                     PrepareForNet(),
                                 ] + ([Crop(size[0])] if self.mode == 'train1' else []))

    @staticmethod
    def images_list_generator(self,images_path):
        images_list = []
        depth_list = []
        games_list = [os.path.join(images_path, game) for game in os.listdir(images_path)]
        for game in games_list:
            dirs = [os.path.join(game, path) for path in os.listdir(game)]
            for dir in dirs:
                if os.path.isdir(dir):
                    data = [os.path.join(dir, path) for path in os.listdir(dir)]
                    for d in data:
                        if 'color' in d:
                            images_list += [os.path.join(d,path) for path in os.listdir(d) if '_' not in path]
                        elif 'depth' == d.split('/')[-1]:
                            depth_list += [os.path.join(d,path) for path in os.listdir(d) if '_' not in path]
                        else:
                            continue
        return images_list, depth_list


    def __getitem__(self, item):
        img_path = self.images_path[item]
        depth_path = img_path.replace('color', 'depth')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # cm to m

        #depth = (depth - depth.min())/(depth.max() - depth.min())
        #depth = depth * 65535 / 100

        sample = self.transform({'image': image, 'depth': depth if self.mode == 'train' else depth[:,:,0]})

        #sample['depth'] = (sample['depth'] - sample['depth'].min()) / (sample['depth'].max() - sample['depth'].min())
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth']) #*255

        sample['valid_mask'] = (sample['depth'] <=250)

        sample['image_path'] = self.images_path[item]

        return sample

    def __len__(self):
        return len(self.images_path)