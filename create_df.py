"""
module that provides 
    dataset creation
    dataset splitting
    metadata modification
"""


import glob
from pathlib import Path
from utils import load_from_json
import pandas as pd
import random
from typing import List, Union
from vk_parser import create_folder
import os
import shutil
import csv


def count_stat(df: pd.core.frame.DataFrame) -> List[int]:
    """
    gives amount of pictures per each class

    input:
        df - dataframe of a pandas type
    output:
        list of ints with the calculated stat
    """

    df.columns = ['file_name', 'label']
    rng = max(df['label']) + 1
    totals = [sum(df['label'] == value) for value in range(rng)]

    return totals


def get_filenames(path: str, extensions: list = ['jpg']) -> list:
    """
    gives all image names in the given directory

    input:
        path - path to the images directory
        extension - the list with valid extensions to look for
    output:
        list with image names
    """

    file_names = []
    for extension in extensions:
        file_names += sorted(glob.glob(path + f"/*.{extension}*"))

    image_names = [Path(value).name for value in file_names]

    return image_names


def separate_images(csv_path: str,
                    dataset_path: str,
                    extensions: list = ['jpg'],
                    specified: bool = False,
                    keys_map_path: Union[dict, None] = None) -> List[str]:
    """
    seprates images per each film type
    input:
        csv_path - path to the csv file
        dataset_path - path to the dataset
        extensions - type of a file, for instance jpg or png or both
        specified - need to separate directly?
        keys_map_path - dict with film types
    output:
        list of filmn names; Neither list[str] or list[list[str]]
    """

    # get filenames
    file_names: list = get_filenames(dataset_path, extensions)

    # separate
    if specified and keys_map_path:

        # load film names
        keys_map: list = load_from_json(keys_map_path)['names']
        film_names = [
            film_name for film in keys_map for film_name in film.keys()]

        # read and find class with min amount of images
        df = pd.read_csv(csv_path)
        sample_range: int = min(count_stat(df))

        sep = []
        for key in film_names:

            key = key.replace(' ', '_')
            names_by_film = [image for image in file_names if key in image]
            sep.append(names_by_film)

        file_names = [random.sample(images, sample_range) for images in sep]
        return file_names

    else:
        return file_names


def move_files(file_names: list, destination: str, path: str = None) -> None:
    """
    moves files to the given dir from the given dir
    input:
        file_names - what to move
        destiantion - where to move
        path - from which dir to move
    """

    for file in file_names:

        if path is not None:

            file_name = os.path.join(path, file)
            shutil.copy2(file_name, destination)

        else:
            shutil.copy2(file, destination)


def split_df(df_path: str,
             separated_names: Union[List[List[str]], List[str]],
             ratio: float,
             train_path: str = None,
             test_path: str = None) -> None:
    """
    splits df to the train and test dataset:
    input:
        df_path - path to the images dataset
        separated_names - what to split
        ratio - train/test ratio
        train_path - where to store the train part
        test_path - where to store the test path
    """

    # make paths
    if train_path is None:
        train_path = os.path.join(df_path, 'train_dataset')

    if test_path is None:
        test_path = os.path.join(df_path, 'test_dataset')

    # make folders
    create_folder(train_path)
    create_folder(test_path)

    # split if df has been separated by min value by the each type of a film
    if isinstance(separated_names[0], List):

        for film_type in separated_names:

            train = random.sample(film_type, int(len(film_type) * ratio))
            test = [file for file in film_type if file not in train]

            move_files(train, train_path, df_path)
            move_files(test, test_path, df_path)

    # without strict separation
    else:

        train = random.sample(separated_names, int(
            len(separated_names) * ratio))
        test = [file for file in separated_names if file not in train]

        move_files(train, train_path, df_path)
        move_files(test, test_path, df_path)


def add_class_to_meta(csv_path: str, file_names: list, class_num: int) -> None:
    """
    adds information to the metadata of a dataset
    input:
        csv_path - path to the csv file
        file_names - file_names
        class_num - which label to add, for instance 1
    """

    with open(csv_path, 'a', encoding='UTF8', newline='') as csv_file:

        for file_name in file_names:

            df_row = [file_name, class_num]
            writer = csv.writer(csv_file)
            writer.writerow(df_row)


def main():

    path = 'D:\\pet_projects\\film_classifier\\dataset'
    csv_path = 'D:\\pet_projects\\film_classifier\\dataset\\df_meta.csv'
    json_path = 'D:\\pet_projects\\film_classifier\\needed_film_names.json'
    # addition_class_path = 'D:\\pet_projects\\.div2k\\images\\DIV2K_train_HR'
    addition_class_path = 'D:\\pet_projects\\film_classifier\\cinestill'

    ratio = 0.8
    class_num = 1

    addittional_file_names = get_filenames(
        addition_class_path, extensions=['png', 'jpg'])
    add_class_to_meta(csv_path, addittional_file_names, class_num)
    move_files(addittional_file_names, path, addition_class_path)

    separated_images = separate_images(csv_path=csv_path, dataset_path=path, extensions=[
                                       'jpg', 'png'], keys_map_path=json_path)
    split_df(path, separated_images, ratio)


# tests
if __name__ == '__main__':
    main()
