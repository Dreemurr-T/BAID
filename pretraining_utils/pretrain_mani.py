import os
import numpy as np
import cv2 as cv
from operations import *
import random
from tqdm import tqdm
import pandas as pd

original_dir = 'pretrain_data/ori'
opt1_dir = 'pretrain_data/opt1/ori'
opt2_dir = 'pretrain_data/opt2/ori'
opt3_dir = 'pretrain_data/opt3/ori'

trp11_dir = 'pretrain_data/opt1/trp1'
trp12_dir = 'pretrain_data/opt1/trp2'
trp21_dir = 'pretrain_data/opt2/trp1'
trp22_dir = 'pretrain_data/opt2/trp2'
trp31_dir = 'pretrain_data/opt3/trp1'
trp32_dir = 'pretrain_data/opt3/trp2'

dirs = [original_dir, opt1_dir, opt2_dir, opt3_dir, trp11_dir, trp12_dir, trp21_dir, trp22_dir, trp31_dir, trp32_dir]


def make_output_dirs():
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def opt_manis(original_image, image_mixup, opt1, opt2, opt3):
    degraded_image = image_manipulation(original_image, opt1, image_mixup)
    degraded_image_2 = image_manipulation(original_image, opt2, image_mixup)
    degraded_image_3 = image_manipulation(original_image, opt3, image_mixup)

    return degraded_image, degraded_image_2, degraded_image_3


def opt_trp(original_image, image_mixup, trp1, trp2):
    degraded_trp = image_manipulation(original_image, trp1, image_mixup)
    degraded_trp2 = image_manipulation(original_image, trp2, image_mixup)

    return degraded_trp, degraded_trp2


def process_data():
    df = pd.read_csv('pretraining_utils/operations.csv')

    file_list = df['original'].values.tolist()
    opts1 = df['opt1'].values.tolist()
    opts2 = df['opt2'].values.tolist()
    opts3 = df['opt3'].values.tolist()

    record = []

    for i in tqdm(range(len(file_list))):
        file_name = file_list[i]
        file_path = os.path.join('images', file_name)
        rand_file = 'images/' + random.choice(file_list)

        opt = opts1[i]
        opt_2 = opts2[i]
        opt_3 = opts3[i]

        original_image = cv.imread(file_path)
        original_image = cv.resize(original_image, (224, 224))
        image_mixup = cv.imread(rand_file)
        image_mixup = cv.resize(image_mixup, (224, 224))

        degraded_image, degraded_image_2, degraded_image_3 = opt_manis(original_image, image_mixup, opt, opt_2, opt_3)

        if opt <= 26:
            if opt % 3 == 0:
                trp1 = opt + 1
                trp2 = opt + 2
                degraded_trp, degraded_trp2 = opt_trp(original_image, image_mixup, trp1, trp2)
            elif opt % 3 == 1:
                trp1 = opt - 1
                trp2 = opt + 1
                degraded_trp, degraded_trp2 = opt_trp(original_image, image_mixup, trp1, trp2)
            elif opt % 3 == 2:
                trp1 = opt - 2
                trp2 = opt - 1
                degraded_trp, degraded_trp2 = opt_trp(original_image, image_mixup, trp1, trp2)
        else:
            degraded_trp = degraded_image
            degraded_trp2 = degraded_image

        if opt_2 <= 26:
            if opt_2 % 3 == 0:
                trp1 = opt_2 + 1
                trp2 = opt_2 + 2
                degraded2_trp, degraded2_trp2 = opt_trp(original_image, image_mixup, trp1, trp2)
            elif opt_2 % 3 == 1:
                trp1 = opt_2 - 1
                trp2 = opt_2 + 1
                degraded2_trp, degraded2_trp2 = opt_trp(original_image, image_mixup, trp1, trp2)
            elif opt_2 % 3 == 2:
                trp1 = opt_2 - 2
                trp2 = opt_2 - 1
                degraded2_trp, degraded2_trp2 = opt_trp(original_image, image_mixup, trp1, trp2)
        else:
            degraded2_trp = degraded_image_2
            degraded2_trp2 = degraded_image_2

        if opt_3 <= 26:
            if opt_3 % 3 == 0:
                trp1 = opt_3 + 1
                trp2 = opt_3 + 2
                degraded3_trp, degraded3_trp2 = opt_trp(original_image, image_mixup, trp1, trp2)
            elif opt_3 % 3 == 1:
                trp1 = opt_3 - 1
                trp2 = opt_3 + 1
                degraded3_trp, degraded3_trp2 = opt_trp(original_image, image_mixup, trp1, trp2)
            elif opt_3 % 3 == 2:
                trp1 = opt_3 - 2
                trp2 = opt_3 - 1
                degraded3_trp, degraded3_trp2 = opt_trp(original_image, image_mixup, trp1, trp2)
        else:
            degraded3_trp = degraded_image_3
            degraded3_trp2 = degraded_image_3

        original_path = os.path.join(original_dir, file_name)

        opt1_path = os.path.join(opt1_dir, file_name)
        opt2_path = os.path.join(opt2_dir, file_name)
        opt3_path = os.path.join(opt3_dir, file_name)

        trp11_path = os.path.join(trp11_dir, file_name)
        trp12_path = os.path.join(trp12_dir, file_name)
        trp21_path = os.path.join(trp21_dir, file_name)
        trp22_path = os.path.join(trp22_dir, file_name)
        trp31_path = os.path.join(trp31_dir, file_name)
        trp32_path = os.path.join(trp32_dir, file_name)

        cv.imwrite(original_path, original_image)

        cv.imwrite(opt1_path, degraded_image)
        cv.imwrite(opt2_path, degraded_image_2)
        cv.imwrite(opt3_path, degraded_image_3)

        cv.imwrite(trp11_path, degraded_trp)
        cv.imwrite(trp21_path, degraded2_trp)
        cv.imwrite(trp31_path, degraded3_trp)

        cv.imwrite(trp12_path, degraded_trp2)
        cv.imwrite(trp22_path, degraded2_trp2)
        cv.imwrite(trp32_path, degraded3_trp2)

        tmp = [original_path, opt1_path, opt2_path, opt3_path, trp11_path, trp12_path, trp21_path, trp22_path, trp31_path, trp32_path]

        record.append(tmp)

    name = ['origin', 'opt1', 'opt2', 'opt3', 'trp1_1', 'trp1_2', 'trp2_1', 'trp2_2', 'trp3_1', 'trp3_2']
    df = pd.DataFrame(columns=name, data=record)
    df.to_csv('pretraining_utils/train.csv', index=False)


if __name__ == '__main__':
    make_output_dirs()
    process_data()

