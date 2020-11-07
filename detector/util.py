from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=False):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    # print(prediction.size(),stride,inp_dim,grid_size)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    prediction[:, :, 5:5 + num_classes] = torch.sigmoid(prediction[:, :, 5:5 + num_classes])
    prediction[:, :, :4] *= stride
    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    # print('in write results')
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    # 计算box的左上角和右下角坐标
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False

    # 一个batch里有多张图片，而每张图片的物体个数都不同，因此必须要循环分别分别处理每一张图片
    for i in range(batch_size):
        img_pred = prediction[i]

        # 每一个box有85个值，其中80个值是物体类别的预测概率，我们选出最大的那个,把80个预测概率删除,然后把所属类别和最大概率拼接在每一行的后面
        max_conf, max_conf_score = torch.max(img_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (img_pred[:, :5], max_conf, max_conf_score)
        img_pred = torch.cat(seq, 1)

        # 去除没有物体的box
        non_zero_ind = torch.nonzero(img_pred[:, 4])
        try:
            image_pred_ = img_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index

        # NMS
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            num = image_pred_class.size(0)  # Number of detections

            for j in range(num):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                # 因为我们要进行idx轮循环，每轮循环可能会删除一些box，最后可能出现索引值j已经比len(image_pred_class)还要大，所以这里用try catch
                try:
                    # 计算第j个box和第j+1到最后一个box的iou
                    ious = bbox_iou(image_pred_class[j].unsqueeze(0), image_pred_class[j + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # 移除IoU > threshold的box(预测同一个物体的box)
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[j + 1:] *= iou_mask

                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(i)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
        # print('write results return shape:'+str(output.size()))
    try:
        return output
    except:
        return 0


def bbox_iou(box1, box2):
    # 两个box的左上和右下角坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def letterbox_img(img, inp_dim):
    """
    lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充,调整图像尺寸
    具体而言,此时某个边正好可以等于目标长度,另一边小于等于目标长度
    将缩放后的数据拷贝到画布中心,返回完成缩放
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim  # inp_dim是需要resize的尺寸（如416*416）
    # 取min(w/img_w, h/img_h)这个比例来缩放，缩放后的尺寸为new_w, new_h,即保证较长的边缩放后正好等于目标长度(需要的尺寸)，另一边的尺寸缩放后还没有填充满.
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    # 将图片按照纵横比不变来缩放为new_w x new_h，768 x 576的图片缩放成416x312.,用了双三次插值
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # 创建一个画布, 将resized_image数据拷贝到画布中心。
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)  # 生成一个我们最终需要的图片尺寸h * w * 3的array,这里生成416x416x3的array,每个元素值为128
    # 将w * h * 3的array中对应new_w * new_h * 3的部分(这两个部分的中心应该对齐)赋值为刚刚由原图缩放得到的数组,得到最终缩放后图片
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_img(img, inp_dim):
    """
    将图像转换为网路的输入格式
    """
    img = (letterbox_img(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(namefile):
    with open(namefile, 'r') as fp:
        names = fp.read().split('\n')[:-1]
    return names