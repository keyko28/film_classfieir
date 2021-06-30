"""
module for parsing vk groups with images

about:
    registration required
    standalone blank app created (for token and version; requires 5 mouse clicks)
    5000 requests per day
    copy a part of an adress bar after https://vk.com/... <- to get a domain

more info about an api:
https://vk.com/dev
"""


from multiprocessing.queues import Queue
import requests
from utils import clearify, load_from_json, create_folder, create_csv, get_film_names
from typing import List, Union
from multiprocessing import Process, Queue
import psutil
import os
import requests
import csv


def check_validity(names: Union[list, set], data: Union[list, set]) -> bool:
    """
    checks is the data valid
    input:
        names - film types
        data - list or set of string from the request
    output:
        checker result: True or False
    """

    checker: int = 0
    setattr(check_validity, 'film_type', None)  # set attr to the function

    for substring in data:

        res: int = 0
        for film_type in names:

            if any(value in substring for film_options in film_type.values()
                   for value in film_options):

                res += 1
                name = [name for name in film_type.keys()]
                # need later to trace cheked film type
                setattr(check_validity, 'film_type', name[0])

        # if more than 1 film name is in the given data
        if res not in (0, 1):
            return False
        else:
            checker += res

    result = True if checker == 1 else False

    if not result:
        setattr(check_validity, 'film_type', None)

    return result


def process_data(data: Union[dict, str], key: str = None) -> Union[list, set]:
    """
    process and clearify data
    input:
        data - dict or str with request
        key - given key
    output:
        list or set of strs
    """

    if isinstance(data, dict):

        if key is None or key not in data.keys():
            raise KeyError('there is no key you are looking for')

        data = data[key]
        data = clearify(data)
        data = data.splitlines()
        data = map(lambda x: x.strip(), data)
        data = map(lambda x: x.lower(), data)
        data = filter(None, data)
        return set(list(data))

    elif isinstance(data, str):
        data = clearify(data)
        data = data.strip().lower()
        return [data]

    else:
        raise TypeError('Only dict or single str are expected')


def get_urls(token: str,
             version: str,
             domain: str,
             film_urls_dict: dict,
             film_names: dict,
             needed_films: list,
             queue: Queue,
             offset: int = 0,
             max_requests_amount: int = 1,
             max_offset: int = 100,
             max_count: int = 100) -> None:
    """
    returns urls from the request
    works using multiprocessing

    input:
        token - authorization token 
        version - api version
        domain - group to look for
        film_urls_dict - empty dict of form of {film type: []}
        film_names -film names
        needed_films - which film types to trace
        queue: Queue to put into
        offset - requlates offset within a span to get a needed posts
        max_requests_amount - how many request to make
        max_offset - max offest to wirk with
        max_count - count of posts per request
    output:
        dict with film type form of: {film type: [urls]}
    """

    for _ in range(max_requests_amount):

        response = requests.get('https://api.vk.com/method/wall.get',
                                params={
                                    'access_token': token,
                                    'v': version,
                                    'domain': domain,
                                    'count': max_count,  # per request
                                    'offset': offset
                                })

        offset += max_offset  # next nth

        data = response.json()['response']['items']
        for item in data:

            try:

                res = process_data(data=item, key='text')
                valid = check_validity(names=film_names, data=res)

                if valid:

                    film = getattr(check_validity, 'film_type', None)

                    if film in needed_films:

                        urls_of_film_type = []
                        attachmets: dict = item['attachments']
                        for attach in attachmets:

                            if attach['type'] == 'photo':
                                photo = attach['photo']
                                bigest_photo = photo['sizes'][-1]
                                urls_of_film_type.append(bigest_photo['url'])

                        if film is not None:
                            film_urls_dict[film] += urls_of_film_type

            except:
                continue

    # this is a return
    if film_urls_dict:
        queue.put(film_urls_dict)
    else:
        queue.put(None)  # strongly recomended to avoid dead procs


def parallelize_url_getting(token: str,
                            version: str,
                            domain: str,
                            film_urls_dict: dict,
                            film_names: dict,
                            needed_films: list,
                            max_requests_amount: int = 1,
                            max_offset: int = 100,
                            max_count: int = 100,
                            logical: bool = False) -> List[dict]:
    """
    setup for get urls using mutliprocessing and queues

    input:
        token - authorization token 
        version - api version
        domain - group to look for
        film_urls_dict - empty dict of form of {film type: []}
        film_names -film names
        needed_films - which film types to trace
        max_requests_amount - how many request to make
        max_offset - max offest to wirk with
        max_count - max amount post per request
        logical - restircs or not uisng logical cores during the operation
    output:
        list with dict of form {fylm type: [urls]}
    """

    # define cpu_qount and sequence
    cpu_count = psutil.cpu_count(logical=logical)
    # offset sequence of form (from, to)
    # gives to each process a needed offset range
    offset_seq = [max_requests_amount *
                  max_offset * cpu for cpu in range(cpu_count)]

    # create processes with a target funtion
    queue = Queue()
    processes = []
    for offset in offset_seq:
        p = Process(target=get_urls, kwargs={'token': token,
                                             'version': version,
                                             'domain': domain,
                                             'film_urls_dict': film_urls_dict,
                                             'film_names': film_names,
                                             'offset': offset,
                                             'max_requests_amount': max_requests_amount,
                                             'max_offset': max_offset,
                                             'max_count': max_count,
                                             'needed_films': needed_films,
                                             'queue': queue})
        processes.append(p)

    for p in processes:
        # allows to automatically kill children after the end of the
        # __main__
        p.daemon = True
        p.start()

    urls = [queue.get() for _ in processes]  # gather form the queue

    for p in processes:
        # kill all
        p.join()

    #  just in case. Kill process if this task has been failed previously
    for p in processes:
        p.kill()

    del queue  # just in case. works more stable

    return [url for url in urls if url]


def unite_urls(urls: List[dict]) -> dict:
    """
    unites urls after collecting
    input:
        urls - given urls
    output:
        dict of form {film type [urls]}
    """

    if not urls:
        raise ValueError(
            'Empty list has been passed. Expected List[dict] instead')

    elif len(urls) == 1:
        return urls[0]

    else:

        to_insert: dict = urls[0]
        from_insert: list = urls[1:]

        for url_dict in from_insert:
            for key, value in url_dict.items():
                for url in value:
                    if url not in to_insert[key]:
                        to_insert[key].append(url)

        return to_insert


def download(request_number: int,
             path: str,
             film_urls: dict,
             csv_filename: str,
             needed_film_names: list,
             needed_only: bool = False) -> None:
    """
    download images from a lis of urls
    inputs:
        request_number - specifies current request*
        path - where to save an image
        film_urls - dict of gathered urls to download from
        csv_filename - name of a csv file
        needed film names - list with film names to look after
        needed_only - regualtes behaviour  for types of film (all/specified)

    *one may want to gather a big dataset, however, restriction of 5000 request per day extists
    to avoid mess with names and csv file, this parameter should be specified directly
    """
    create_folder(path)
    create_csv(path, csv_filename)
    csv_path = os.path.join(path, csv_filename)

    if needed_only:

        film_urls = {key: value for key, value in film_urls.items() if value}
        for film, urls in film_urls.items():

            for num, url in enumerate(urls):

                raplaced = film.replace(' ', '_')  # clearify name
                image_name = [str(request_number), raplaced, str(num)]
                image_name = '_'.join(image_name) + '.jpg'
                image_path = os.path.join(path, image_name)

                if not os.path.isfile(image_path):  # ignore if already downloaded
                    response = requests.get(url, stream=True)

                    # save image
                    with open(image_path, 'wb') as outfile:
                        outfile.write(response.content)

                    # save metadata
                    image_class = needed_film_names.index(film)
                    df_row = [image_name, image_class]
                    with open(csv_path, 'a', encoding='UTF8', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(df_row)

    else:

        for image_class, (film, urls) in enumerate(film_urls.items()):

            for num, url in enumerate(urls):

                raplaced = film.replace(' ', '_')
                image_name = [str(request_number), raplaced, str(num)]
                image_name = '_'.join(image_name) + '.jpg'
                image_path = os.path.join(path, image_name)

                if not os.path.isfile(image_path):  # ignore if already downloaded
                    response = requests.get(url, stream=True)

                    # save image
                    with open(image_path, 'wb') as outfile:
                        outfile.write(response.content)

                    # save metadata
                    df_row = [image_name, image_class]
                    with open(csv_path, 'a', encoding='UTF8', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(df_row)


def main() -> None:

    test_path = 'D:\\pet_projects\\film_classifier\\all_film_names.json'
    needed_film_names_path = 'D:\\pet_projects\\film_classifier\\needed_film_names.json'
    df_path = 'D:\\pet_projects\\film_classifier\\dataset'
    film_names = get_film_names(test_path=test_path)
    needed_film_names = (get_film_names(test_path=needed_film_names_path))
    needed_film_names = [
        key for film_type in needed_film_names for key in film_type.keys()]

    token = '4b03db604b03db604b03db60cd4b7bd5ba44b034b03db602bb761b83b0509562b9b5af7'
    version = 5.131
    domain = 'trita.plenka'

    max_requests_amount: int = 200
    max_offset: int = 100
    max_count: int = 100

    film_urls_by_name = [
        name for film_type in film_names for name in film_type.keys()]
    film_urls_by_name = {key: [] for key in film_urls_by_name}

    urls = parallelize_url_getting(token=token,
                                   version=version,
                                   domain=domain,
                                   film_urls_dict=film_urls_by_name,
                                   max_requests_amount=max_requests_amount,
                                   max_offset=max_offset,
                                   max_count=max_count,
                                   film_names=film_names,
                                   needed_films=needed_film_names)

    urls = unite_urls(urls)
    download(request_number=0,
             path=df_path,
             film_urls=urls,
             csv_filename='df_meta.csv',
             needed_film_names=needed_film_names,
             needed_only=True)


if __name__ == '__main__':
    main()
