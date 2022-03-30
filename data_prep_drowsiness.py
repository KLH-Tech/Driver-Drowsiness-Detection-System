import os
import glob
import shutil
import random
from tqdm import tqdm

raw_data = r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\mrlEyes_2018_01'
for dirpath, dirname, filename in os.walk(raw_data):
    for file in tqdm([f for f in filename if f.endswith('.png')]):
        if file.split('_')[4] == '0':
            path=r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Prepared  Data\Train\Close Eyes'
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copy(src=dirpath + '/' + file, dst= path)
        elif file.split('_')[4] == '1':
            path=r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Prepared  Data\Train\Open Eyes'
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copy(src=dirpath + '/' + file, dst= path)


def create_test_closed(source, destination, percent):
    '''
    divides closed eyes images into given percent and moves from
    source to destination.

    Arguments:
    source(path): path of source directory
    destination(path): path of destination directory
    percent(float): percent of data to be divided(range: 0 to 1)
    '''
    path, dirs, files_closed = next(os.walk(source))
    file_count_closed = len(files_closed)
    percentage = file_count_closed * percent
    to_move = random.sample(glob.glob(source + "/*.png"), int(percentage))

    for f in enumerate(to_move):
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.move(f[1], destination)
    print(f'moved {int(percentage)} images to the destination successfully.')


def create_test_open(source, destination, percent):
    '''
    divides open eyes images into given percent and moves from
    source to destination.

    Arguments:
    source(path): path of source directory
    destination(path): path of destination directory
    percent(float): percent of data to be divided(range: 0 to 1)
    '''
    path, dirs, files_open = next(os.walk(source))
    file_count_open = len(files_open)
    percentage = file_count_open * percent
    to_move = random.sample(glob.glob(source + "/*.png"), int(percentage))

    for f in enumerate(to_move):
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.move(f[1], destination)
    print(f'moved {int(percentage)} images to the destination successfully.')


create_test_closed(r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Prepared  Data\Train\Close Eyes',
                   r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Prepared  Data\Test\Close Eyes',
                   0.2)

create_test_open(r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Prepared  Data\Train\Open Eyes',
                    r'C:\Users\VISHU_PERI\Desktop\VISHU\KLH_2nd_Year\2nd_Sem\PFSD\PFSD_PROJECT\Prepared  Data\Test\Open Eyes',
                    0.2)