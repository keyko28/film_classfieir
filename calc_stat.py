"""
This module provides instruments to:
    - calc the mean value per each channel of an image
    - calc the std value per each channel of an image 
"""


from PIL import Image, ImageStat
import numpy as np
from glob import glob
import os
from typing import List, Union
from multiprocessing import Pool
from psutil import cpu_count


def get_file_names(path: str, extension: Union[List, str]) -> List:
    """
    returns file_names to work with
    inputs:
        :path - the path to the dataset directory
        :extentions - a list or an sting to work with

    output:
        :list 
    """
    if isinstance(extension, List):

        images: list = []
        for ext in extension:
            request = os.path.join(path, f'*.{ext}')
            images += glob(request)  # to avoid merging opperation after

        return images

    elif isinstance(extension, str):

        request = os.path.join(path, f'*.{extension}')
        return glob(request)  # list

    else:

        raise TypeError('Only list or str are expected')


def img_to_array(img: Image) -> np.ndarray:
    """
    convert an image to an np.array
    input:
        :img - an instance of an Image class
    output:
        :np.ndarray - an array representation of an image

    """
    return np.array(img)


def get_mean(image_path: str) -> list:
    """
    returns a list of mean values per each channel of an image
    input:
        :image_path - path to a file
    output:
        :list of mean values
    """
    image = Image.open(image_path)
    avg = ImageStat.Stat(image).mean

    # 255 for scale between 0 and 1
    return [value / 255 for value in avg]


def get_std(image_path: str) -> list:
    """
    returns a list of std values per each channel of an image
    input:
        :image_path - path to a file
    output:
        :list of std values
    """
    image = Image.open(image_path)
    avg = ImageStat.Stat(image).stddev

    return [value / 255 for value in avg]


def process_images(paths: str, pool: Pool):
    """
    returns a list of std values per each channel of an image
    input:
        :path - path to a file
        :pool - an instance of an Pool class to provide parallel processing
    output:
        :tuple of map objects which contain mean and std values

    *map(sum, zip(*some_iterable) - gives an opportunity to sum over collumns
    if not 3 dim values has been returned than 1 channel images are presented
    """

    lenght = len(paths)

    avg_per_chanel = pool.map(get_mean, paths)
    std_per_chanel = pool.map(get_std, paths)

    if not all(len(value) == 3 for value in avg_per_chanel):

        print([index for index, value in enumerate(
            avg_per_chanel) if len(value) != 3])
        raise ValueError('BW images have been found')

    avg = map(lambda x: x/lenght, map(sum, zip(*avg_per_chanel)))
    std = map(lambda x: x/lenght, map(sum, zip(*std_per_chanel)))

    return avg, std

# use only as an external function in other package


def gather_stat(path: str, extensions: str, logical: bool = False):
    """
    the main function which can be used as callable object in an external module
    returns a list similar to [means, stds]
    input:
        :path - path to a file
        :extentions - a list or an sting to work with
        :logical - should one use only real cores
    output:
        :list of means, stds

    if logical false, than only real cores will be utilized
    choose True for utilizing all cores
    """
    workers = cpu_count(logical=logical)
    pool = Pool(workers)

    res = get_file_names(path, extensions)
    rs = process_images(res, pool)

    pool.close()
    pool.join()

    return [list(value) for value in rs]


if __name__ == '__main__':

    # demo use
    workers = cpu_count(logical=False)
    pool = Pool(workers)

    # path to the dataset
    res = get_file_names(
        'D:\\pet_projects\\film_classifier\\dataset\\train_dataset', ['jpg', 'png'])
    rs = process_images(res, pool)  # fires

    pool.close()  # process workers
    pool.join()  # workers

    print([list(val) for val in rs])  # display results
