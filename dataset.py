"""
dataset module
"""


import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import List
import pandas as pd
from pathlib import Path
import csv
import numpy as np


class ImageDataset(Dataset):
    """
    ended up with two options:
        pd dataframe - sometimes unstable, but strict
        dict - creates dict from csv - less explicit, more stable
    """

    def __init__(self,
                 path: str,
                 csv_path: str,
                 mean: List[float],
                 std: List[float],
                 need_dict: bool = False,
                 train: bool = True) -> None:
        """
        input:
            path - path to the direcotry contains images, 
            csv_path - path to the file with metadata for the dataset
            mean - value wich is known or gotten by the 'calc_stat.py' module
            std - value wich is known or gotten by the 'calc_stat.py' module
            need_dict - defines type of return
            train - defines type of dataset  
        """

        self.files = sorted(glob.glob(path + "/*.jpg*"))
        self.metadata = pd.read_csv(csv_path)
        # add column names to the df
        self.metadata.columns = ['file_name', 'label']
        self.need_dict = need_dict
        self.transform = None

        if train:

            # define transforms
            self.transform = transforms.Compose([transforms.Resize(256),
                                                 transforms.RandomResizedCrop(
                                                     224),
                                                 transforms.RandomRotation(25),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])

        else:

            # define transforms
            self.transform = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])

        if self.need_dict:  # creates dict to work with

            with open(csv_path, mode='r') as infile:
                reader = csv.reader(infile)
                dict_from_csv = {rows[0]: rows[1] for rows in reader}

        self.metadata = dict_from_csv

    def __getitem__(self, index: int) -> dict:

        idx = index % len(self.files)
        image_name = Path(self.files[idx]).name

        img = Image.open(self.files[idx])
        img = self.transform(img)

        if not self.need_dict:
            label = self.metadata.loc[self.metadata['file_name']
                                      == image_name, 'label'].item()
        else:
            label = int(self.metadata[image_name])

        return {'img': img, 'label': label}

    def __len__(self):

        return len(self.files)


# tests
if __name__ == '__main__':

    mean = np.array(
        [0.45328920380047055, 0.43493361995304664, 0.4093699853405604])
    std = np.array(
        [0.2487343734737629, 0.23233310286204029, 0.22754716138014985])

    path = ''  # to the dataset
    csv_path = ''  # to the csv
    train_dataset = ImageDataset(
        path=path, csv_path=csv_path, mean=mean, std=std, need_dict=True)

    print(train_dataset[0])
