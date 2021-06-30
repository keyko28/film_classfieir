"""
moduel with useful utils functions
"""

from typing import Any, List
import json
import regex as re
from PIL import Image
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re


def get_film_names(test_path: str, key: str = 'names') -> List[dict]:
    """
    require film names from json file:
    input:
        test_path - path to the json file
        key - key to make query with
    output:
        list of dicts with info per each film type
    """

    names = load_from_json(test_path)
    return names[key]


def clearify(text: str) -> str:
    """
    get rid of unuseful text symbols in the given str
    input:
        text - given str
    output:
        clearifyed str
    """

    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                r"\?|\.|\!|\/|\;|\:|\*|\&|\%|\#|\@|\+|\$"
                                "]+", flags=re.UNICODE)

    return regrex_pattern.sub(r'', text)


def load_from_json(path: str) -> dict:
    """
    function to load data from json and return dict
    input:
        path - path to the file
    output:
        dict with needed data
    """

    with open(path, encoding='utf-8') as json_file:
        data = json.load(json_file)

    return data


def proccess_film_names(names: dict, key: str = None) -> set:
    """
    function to process film names
    returns set of unique names
    all leters are lower
    spaces will have been removed

    input:
        names - a dict from json with film names
        key - a key to get list with names

    output:
        a set with preprocessed names from list  
    """

    # checkers
    if key is None:
        raise ValueError('key must be specified explicitly')

    if key not in names.keys():
        raise KeyError('there is no such key you are looking for')

    names_lst: List[str] = names[key]

    # process names:
    names_lst = map(lambda x: x.strip(), names_lst)
    names_lst = map(lambda x: x.lower(), names_lst)

    return set(list(names_lst))


def create_folder(path: str) -> None:
    """
    creates folder in needed diractory
    inputs:
        :path - where should we create a direcotry
    """

    if not os.path.isdir(path):
        os.makedirs(path)


def create_csv(path: str, name: str) -> None:
    """
    creates emty csv in needed diractory
    inputs:
        :path - where should we create a direcotry
        :name - name of the file
    """

    full_filename = os.path.join(path, name)

    if not os.path.exists(full_filename):
        with open(full_filename, 'w') as empty_csv:
            pass


def check_bw_images(path: str) -> list:
    """
    checks for black and white images in the directory
    input:
        path - str with path to the directory to search in
    output:
        list with bw images
    """

    files = (file for ext in ['jpg', 'png']
             for file in glob.glob(path + f"/*.{ext}*"))

    result = []
    for image in files:
        dims = np.array(Image.open(image)).shape[-1]

        if dims != 3:
            result.append(image)

    return result


def calc_weights(df_path: str, csv_path: str) -> dict:
    """
    calcs weights of a class
    input:
        df_path - path to the dataset
        csv_path - path to the metadata
    output:
        dict with weights per each class
    """

    df = pd.read_csv(csv_path, delimiter=',')
    df.columns = ['file_name', 'label']

    files = []
    for ext in ['jpg', 'png']:
        files += sorted(glob.glob(df_path + f"/*.{ext}*"))

    image_names = [Path(value).name for value in files]
    length = len(image_names)

    keys = list(range(max(df['label']) + 1))
    result = {key: 0 for key in keys}

    for image in image_names:
        try:
            label = df.loc[df['file_name'] == image, 'label'].item()
            result[label] += 1
        except:
            continue

    for key, value in result.items():
        result[key] = value/length

    return result


#tests and examples
if __name__ == '__main__':

    res = check_bw_images('D:\\pet_projects\\film_classifier\\dataset')
    print(res)

    path = 'D:\\pet_projects\\film_classifier\\dataset\\train_dataset'
    csv_path = 'D:\\pet_projects\\film_classifier\\dataset\\df_meta.csv'
    print(calc_weights(path, csv_path))
