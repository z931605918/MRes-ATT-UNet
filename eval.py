from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import shutil
import random
from Models import  U_Net,AttU_Net,MultiResUNet,MR_Att_Unet_1
from losses import calc_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time
def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))
class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir,transformI = None, transformM = None):
        self.images = os.listdir(images_dir)
        self.images.sort(key=lambda x:int(x[:-4]))
        self.labels = os.listdir(labels_dir)
        self.labels.sort(key=lambda x:int(x[:-4]))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM
        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512,512)),

                torchvision.transforms.ToTensor(),

            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512,512)),


                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),

            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        #print(self.labels[i])
        i1 = Image.open(self.images_dir + '\\'+ self.images[i]).convert('RGB')
        l1 = Image.open(self.labels_dir +'\\' + self.labels[i]).convert('RGB')
        img = self.tx(i1)
        label = self.lx(l1)
        return img, label


def test_model(model, para_names, save_file_name, dataloader, valid_dataloader):
    model_test_1 = model_unet(model, 3, 1)
    trloss_line_unet = []
    teloss_line_unet = []
    for i in range(20, 501, 20):
        parameters_1 = torch.load(' '.format(para_names, i))
        model_test_1.load_state_dict(parameters_1['model_state_dict'])
        model_test_1 = model_test_1.eval()
        model_test_1.cuda()
        loss_tr = 0.0
        for x, y in dataloader:
            with torch.no_grad():
                loss_C1 = 0.0
                loss_S1 = 0.0
                loss_1 = 0.0
                x = x.cuda()

                outputs_1 = model_test_1(x).cpu()
                loss_C1 = F.binary_cross_entropy_with_logits(outputs_1, y)
                outputs_2 = F.sigmoid(outputs_1)
                loss_S1 = dice_loss(outputs_2, y)
                loss_1 = (loss_C1 + loss_S1) / 2
                loss_tr += loss_1

                print(loss_C1, loss_S1)
        loss_tr = loss_tr / len(dataloader)
        trloss_line_unet.append(loss_tr)
        print('Trainingset')
        print(i)
        print(trloss_line_unet)

        loss_te = 0.0
        for x, y in valid_dataloader:
            with torch.no_grad():
                loss_C1 = 0.0
                loss_S1 = 0.0
                loss_1 = 0.0
                x = x.cuda()

                outputs_1 = model_test_1(x).cpu()
                loss_C1 = F.binary_cross_entropy_with_logits(outputs_1, y)
                outputs_2 = F.sigmoid(outputs_1)
                loss_S1 = dice_loss(outputs_2, y)
                loss_1 = (loss_C1 + loss_S1) / 2
                loss_te += loss_1
            # print(loss_C1,loss_S1)
        loss_te = loss_te / len(valid_dataloader)
        teloss_line_unet.append(loss_te)
        print('Testset')
        print(i)
        print(teloss_line_unet)
        torch.cuda.empty_cache()
    loss_line_1 = {'tr': trloss_line_unet, 'te': teloss_line_unet}
    torch.save(loss_line_1, ''.format(save_file_name))  #save path
    return trloss_line_unet, teloss_line_unet

