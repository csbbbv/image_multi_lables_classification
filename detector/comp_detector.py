from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2

import argparse
import os
import os.path as osp
import pickle as pkl
import pandas as pd
import random

from darknet import Darknet
from util import *

# from dectetor.util import load_classes

colors = pkl.load(open('D:\dev\code\python\myyolo\dectetor\pallete', 'rb'))


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


# x shape:[id, x1, y1, x2, y2, objectness score, the score of class with maximum confidence, index of class]
def write_box(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    color = random.choice(colors)
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 2)
    return img


def get_box_coordinates(args):
    images = args["images"]
    batch_size = int(args["bs"])
    confidence = float(args["confidence"])
    nms_thesh = float(args["nms_thresh"])
    start = 0
    CUDA = torch.cuda.is_available()
    if CUDA:
        print('cuda is available')
    else:
        print('cuda is unavailable')

    num_classes = 80
    classes = load_classes('D:\dev\code\python\myyolo\dectetor\coco.names')

    print('loading network...')
    model = Darknet(args["cfgfile"])
    model.load_weights(args["weightsfile"])
    print('network loaded successfully')

    model.net_info['height'] = args["reso"]
    inp_dim = int(model.net_info['height'])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()
    model.eval()

    read_dir = time.time()
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print('No file or directory with the name {}'.format(images))
        exit(0)

    if not os.path.exists(args["det"]):
        os.mkdir(args["det"])

    loaded_ims = [cv2.imread(x) for x in imlist]
    load_batch = time.time()
    # print(len(loaded_ims))
    # PyTorch Variables for images
    im_batches = list(map(prep_img, loaded_ims, [inp_dim for x in range(len(imlist))]))

    # List containing dimensions of original images
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    left_over = 0
    if len(im_dim_list) % batch_size:
        left_over = 1
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + left_over
        im_batches = [torch.cat((im_batches[i * batch_size:min((i + 1) * batch_size, len(im_batches))])) for i in
                      range(num_batches)]

    write = 0
    start_det_loop = time.time()
    print('begin detection...')
    for i, batch in enumerate(im_batches):
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
        # print('size:'+str(prediction.size()))
        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)
        end = time.time()
        # print(prediction.size)
        if type(prediction) == int:
            for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
                im_id = i * batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue

        prediction[:, 0] += i * batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            # print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            # print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            # print("----------------------------------------------------------")
        if CUDA:
            torch.cuda.synchronize()
        try:
            output
        except NameError:
            # print('No detections were made')
            # exit()
            continue
    # print('end detection')

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

    # [id, x1, y1, x2, y2, objectness score, the score of class with maximum confidence, index of class]
    # 将box的坐标变换到未缩放的图片坐标下
    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    # 有些框会超出图片的范围，将这些框的坐标进行修剪
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])
    # print(output.shape)
    ########################################### for competition
    final_write = 0
    for i in range(len(output)):
        if output[i][-1] == 0:
            if not final_write:
                final_output = output[i]
                final_write = 1
            else:
                final_output = np.concatenate((final_output, output[i]))
    final_output = final_output.reshape((-1, 8))

    imgnames = os.listdir(images)



    ###########################################

    book = dict()
    for name in imgnames:
        book[name] = []
    for i in range(len(final_output)):
        book[imgnames[int(final_output[i][0])]].append([float(final_output[i][1]),float(final_output[i][2]),float(final_output[i][3]),float(final_output[i][4])])

    # print(book)
    torch.cuda.empty_cache()
    return book
    
args = {'images':'./data2','det':'det','bs':1,'confidence':0.5,'nms_thresh':0.4,'cfgfile':'cfg/yolov3.cfg','weightsfile':'yolov3.weights','reso':416}
book = get_box_coordinates(args)
print(book)
