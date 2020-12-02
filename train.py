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
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import shutil
import random
from Models import U_Net,AttU_Net,MultiResUNet,MR_Att_Unet_1
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time


def precies_recall(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0
    with torch.no_grad():
        i_flat = prediction.view(-1).cpu().numpy()
        i_flat = np.where(i_flat < 0.5, 0, 1)
        t_flat = target.view(-1).cpu().numpy()

        intersection = (i_flat * t_flat).sum()

    return (intersection + 0.0001) / (t_flat.sum() + 0.0001), (intersection + 0.0001) / (i_flat.sum() + 0.0001)

#from ploting import VisdomLinePlotter
#from visdom import Visdom
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Setting the basic paramters of the model
#######################################################

batch_size = 2
print('batch_size = ' + str(batch_size))
valid_size = 0.15
epoch = 500
print('epoch = ' + str(epoch))
random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))
shuffle = True
valid_loss_min = np.Inf
num_workers = 4
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0
pin_memory = False
if train_on_gpu:
    pin_memory = True


model_Inputs = [U_Net,AttU_Net,MultiResUNet,MR_Att_Unet_1]




def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test




model_test = model_unet(model_Inputs[2], 3, 1)




inputs_data_1=''
labels_data_1=''
test_image = ''
test_label = ''
test_folderP = ''
test_folderL = ''

Training_Data_1 = Images_Dataset_folder(inputs_data_1,
                                      labels_data_1)
Valid_Data=Images_Dataset_folder(test_folderP,test_folderL)

#######################################################

data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])



num_train = len(Training_Data_1)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))



train_loader_1 = torch.utils.data.DataLoader(Training_Data_1, batch_size=batch_size,# sampler=train_sampler,
                                           shuffle=True,num_workers=num_workers, pin_memory=pin_memory,)
#train_loader_2 = torch.utils.data.DataLoader(Training_Data_2, batch_size=batch_size, sampler=train_sampler,
                                        #   num_workers=num_workers, pin_memory=pin_memory,)
valid_loader = torch.utils.data.DataLoader(Valid_Data, batch_size=batch_size,# sampler=valid_sampler,
                                           shuffle=True,num_workers=num_workers, pin_memory=pin_memory,)


initial_lr = 0.001
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr) # try SGD
#opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)

MAX_STEP = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=4e-6)
#scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)





New_folder = './model'

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)



read_pred = './model/pred'


if os.path.exists(read_pred) and os.path.isdir(read_pred):
    shutil.rmtree(read_pred)

try:
    os.mkdir(read_pred)
except OSError:
    print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
else:
    print("Successfully created the prediction directory '%s' of dice loss" % read_pred)


read_model_path = './model/Unet_D_' + str(epoch) + '_' + str(batch_size)

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

for i in range(epoch):
    model_test.to(device)
    crossen_loss = 0.0
    train_loss = 0.0
    valid_loss = 0.0
    sum_precies = 0.0
    sum_recall = 0.0
    since = time.time()
    scheduler.step(i)
    lr = scheduler.get_lr()
    #######################################################
    # Training Data
    #######################################################
    model_test.train()
    k = 1
    for x, y in train_loader_1:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)  # Dice_loss Used
        train_loss += lossT.item()
        CE_loss = F.binary_cross_entropy_with_logits(y_pred, y)
        crossen_loss += CE_loss.item()
        y_0_1 = F.sigmoid(y_pred)
        precies, recall = precies_recall(y_0_1, y)
        # print(lossT.item(),CE_loss.item(),precies,recall)
        sum_precies += precies
        sum_recall += recall
        # print(lossT.item())
        lossT.backward()
        #  plot_grad_flow(model_test.named_parameters(), n_iter)
        opt.step()
        # y_pred=torch.tensor(y_pred,dtype=torch.long)
        # y=torch.tensor(y,dtype=torch.long)
        # y=y.to(device)
        # x_size = lossT.item() * x.size(0)
        # k = 2
    if i % 1 == 0:
        with torch.no_grad():
            tensor_to_img = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
            image = tensor_to_img(x[0].cpu())
            labels = tensor_to_img(y[0].cpu())
            outputs = model_test(x).cpu()
            outputs = F.sigmoid(outputs[0][0]).numpy()
            outputs1 = np.where(outputs < 0.5, 0, 1)
            # outputs=tensor_to_img(255*outputs).convert('1')

    train_loss = batch_size * train_loss / num_train
    sum_precies = batch_size * sum_precies / num_train
    sum_recall = batch_size * sum_recall / num_train
    crossen_loss = batch_size * crossen_loss / num_train

    Muti_loss_line.append(train_loss)
    Cross_loss_line.append(crossen_loss)
    Precies_line.append(sum_precies)
    Recall_line.append(sum_recall)

    print('Epoch: {}/{} \tTraining Loss: {:.6f} lr:{} '.format(i + 1, epoch, train_loss, lr
                                                               ))
    print('cross entropy loss :{:.6f} , precies :{:.6f} , recall:{:.6f}'.format(crossen_loss, sum_precies, sum_recall))
    if (i + 1) % 20 == 0:
        model_test.eval()
        checkpoint = {'model_state_dict': model_test.state_dict(),
                      'optimizer_state_dict': opt.state_dict(),
                      'epoch': epoch
                      }
        path_checkpoint = '  '.format(i + 1)  #save path
        torch.save(checkpoint, path_checkpoint)

#######################################################
#Checking if GPU is used
#######################################################

