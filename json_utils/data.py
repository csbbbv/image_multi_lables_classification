'''
input : [image_dir,x1,y1,x2,y2]
'''
import os
import time
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torchvision import transforms
from torchvision import models
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from numpy import printoptions
import requests
import tarfile
import random
import json
import  random
import shutil
from shutil import copyfile
from config import *
# 数据集加载
# Dataloader，标签二值化.



def split_data(image_dir,scale=0.7):
    data_list = os.listdir(image_dir)
    length = len(data_list)
    train_num = int(length*scale)
    random.shuffle(data_list)
    train_list = data_list[:train_num]
    test_list = data_list[train_num:]

    return  train_list,test_list

def move_file(file_list,old_dir,new_dir,format='.jpg'):
    for i in file_list:
        file_name = i.split('.')[0]
        shutil.move(old_dir+file_name+format, new_dir)


if __name__=='__main__':
    train_list,test_list = split_data(image_dir)
    move_file(train_list,image_dir,'D:\\pycharm\\data\\train\\image\\','.jpg')
    move_file(train_list, 'D:\\pycharm\\data\\new_clothes\\', 'D:\\pycharm\\data\\train\\anno\\', '.json')
    move_file(test_list, image_dir, 'D:\\pycharm\\data\\test\\image\\', '.jpg')
    move_file(test_list, 'D:\\pycharm\\data\\new_clothes\\', 'D:\\pycharm\\data\\test\\anno\\', '.json')
