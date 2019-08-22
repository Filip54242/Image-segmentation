import random

import numpy as np
import torch
from PIL import Image

from VOC2012colors import OutputMaker
from datasettype import DatasetType
from torchvision import transforms


class Voc2012Dataset(torch.utils.data.Dataset):

    def get_train(self):
        name_list = open(self.root_dir + "/ImageSets/Segmentation/train.txt", 'r').read().split("\n")
        name_list.pop()
        return name_list

    def get_train_validation(self):
        name_list = open(self.root_dir + "/ImageSets/Segmentation/trainval.txt", 'r').read().split("\n")
        name_list.pop()
        return name_list

    def get_validation(self):
        name_list = open(self.root_dir + "/ImageSets/Segmentation/val.txt", 'r').read().split("\n")
        name_list.pop()
        return name_list

    def __init__(self, root_dir, transform, set_type=DatasetType.TRAIN, num_images=-1):
        self.color_classes = OutputMaker()
        self.color_classes.load_image_data()

        self.root_dir = root_dir
        self.transform = transform
        self.data_set = None
        self.num_images = num_images

        if set_type == DatasetType.TRAIN:
            self.data_set = self.get_train()
        elif set_type == DatasetType.TRAIN_VALIDATION:
            self.data_set = self.get_train_validation()
        elif set_type == DatasetType.VALIDATION:
            self.data_set = self.get_validation()
        if num_images > 0:
            self.data_set = random.sample(self.data_set, num_images)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img_name = self.root_dir + "/JPEGImages/" + self.data_set[idx] + ".jpg"
        label_name = self.root_dir + "/SegmentationClass/" + self.data_set[idx] + ".png"

        image = Image.open(img_name)
        target = Image.open(label_name)

        image = self.transform(image)

        transform_resize = transforms.Compose([transforms.Resize((224, 224))])
        target = np.array(transform_resize(target), dtype='int64')
        target[target == 255] = 0

        # target = np.array(target, dtype="int64")
        # target[target == 255] = 0
        # target.resize((224, 224))

        return image, target
