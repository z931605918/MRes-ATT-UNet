from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class up_conv_1(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv_1, self).__init__()
        in_ch=int(int(0.167*1.667*in_ch)+int(0.333*1.667*in_ch)+int(0.5*1.667*in_ch))
        out_ch=int(int(0.167*1.667*out_ch)+int(0.333*1.667*out_ch)+int(0.5*1.667*out_ch))
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out

class res_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(res_block,self).__init__()
        self.res = conv_block(ch_in,ch_out)
        self.main = conv_block(ch_in,ch_out)
        self.bcn=nn.BatchNorm2d(ch_out)

    def forward(self,x):
        res_x = self.res(x)
        main_x = self.main(x)
        out = torch.add(res_x, main_x)
        out = nn.ReLU(inplace=True)(out)
        out = self.bcn(out)
        return out



class ResPath(nn.Module):
    def __init__(self,ch,stage):
        super(ResPath,self).__init__()

        self.stage = stage
        #self.conv_1x1=conv_block(ch,int(0.167*1.667*ch)+int(0.333*1.667*ch)+int(0.5*1.667*ch),kernel_size=1)
        self.block = res_block(int(0.167*1.667*ch)+int(0.333*1.667*ch)+int(0.5*1.667*ch), int(0.167*1.667*ch)+int(0.333*1.667*ch)+int(0.5*1.667*ch))

    def forward(self, x):
        out = self.block(x)
        #x=self.conv_1x1(x)
        for i in range(self.stage-1):
            out = self.block(out)

        return out

class MultiResBlock(nn.Module):
    def __init__(self,in_ch=3,out_ch=1,bias=0):
        super(MultiResBlock, self).__init__()
        W = out_ch
        self.in_ch=in_ch
        self.W=W*1.667
        self.residual_layer = conv_block(int(0.167*1.667*in_ch)+int(0.333*1.667*in_ch)+int(0.5*1.667*in_ch)+bias, int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5), 1, 1, 0)
        self.conv3x3 = conv_block(int(0.167*1.667*in_ch)+int(0.333*1.667*in_ch)+int(0.5*1.667*in_ch+bias), int(self.W * 0.167))
        self.conv5x5 = conv_block(int(self.W * 0.167), int(self.W * 0.333))
        self.conv7x7 = conv_block(int(self.W * 0.333), int(self.W * 0.5))
        self.bcn1=nn.BatchNorm2d(int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5))
        self.bcn2=nn.BatchNorm2d(int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5))

    def forward(self, x):

        res = self.residual_layer(x)
        sbs = self.conv3x3(x)
        obo = self.conv5x5(sbs)
        cbc = self.conv7x7(obo)

        all_t = torch.cat((sbs, obo, cbc), 1)

        all_t_b = self.bcn1(all_t)
        out = torch.add(all_t_b, res)
        out = nn.ReLU(inplace=True)(out)
        out = self.bcn2(out)
        return out

class MultiResUNet(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MultiResUNet,self).__init__()
        self.ch_out=ch_out

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.cconv1=conv_block(ch_in,51)

        self.mresblock1=MultiResBlock(32,32)


        self.back_mres1 = MultiResBlock(64, 32,-3)

        self.mresblock2 = MultiResBlock(32,64)

        self.res_path1 = ResPath(32, 4)
        self.Up2 = up_conv_1(64, 32)
        self.back_mres2 = MultiResBlock(128,64,-2)

        self.mresblock3 = MultiResBlock(64,128)

        self.res_path2 = ResPath(64, 3)
        self.Up3 = up_conv_1(128, 64)
        self.back_mres3 = MultiResBlock(256, 128,-2)

        self.mresblock4 = MultiResBlock(128,256)
        self.res_path3 = ResPath(128, 2)
        self.Up4 = up_conv_1(256, 128)
        self.back_mres4 = MultiResBlock(512, 256)

        self.res_path4 = ResPath(256, 1)
        self.Up5 = up_conv_1(512, 256)
        self.mresblock5 = MultiResBlock(256,512)





        self.out_cov=nn.Conv2d(51,self.ch_out,1,1,0)


    def forward(self, x):
        x1=  self.cconv1(x)
        x1 = self.mresblock1(x1)
        res_x1 = self.res_path1(x1)
        x2 = self.Maxpool(x1)

        x2 = self.mresblock2(x2)
        res_x2 = self.res_path2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.mresblock3(x3)
        res_x3 = self.res_path3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.mresblock4(x4)
        res_x4 = self.res_path4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.mresblock5(x5)

        d4=self.Up5(x5)
        d4=torch.cat((d4,res_x4),dim=1)
        d4=self.back_mres4(d4)

        d3=self.Up4(d4)
        d3=torch.cat((d3,res_x3),dim=1)
        d3=self.back_mres3(d3)

        d2=self.Up3(d3)
        d2=torch.cat((d2,res_x2),dim=1)
        d2=self.back_mres2(d2)

        d1=self.Up2(d2)
        d1=torch.cat((d1,res_x1),dim=1)
        d1=self.back_mres1(d1)
        d1=self.out_cov(d1)

        return d1

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out,psi

class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1,psi=False):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
        self.psi=psi
        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        x4,psi4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3,psi3= self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2,psi2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1,psi1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        if self.psi:
            return psi1,psi2,psi3,psi4
        #  out = self.active(out)

        return out

class MultiResBlock_1(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super(MultiResBlock_1, self).__init__()
        W = out_ch
        self.W=W
        self.residual_layer = conv_block(in_ch, int(W * 0.375) + int(W * 0.625), 1, 1, 0)
        self.conv3x3 = conv_block(in_ch, int(W * 0.375))
        self.conv5x5 = conv_block(int(W * 0.375), int(W * 0.625))
        self.bcn1=nn.BatchNorm2d(int(self.W * 0.375) + int(self.W * 0.625))
        self.bcn2=nn.BatchNorm2d(out_ch)
    def forward(self, x):
        res = self.residual_layer(x)
        sbs = self.conv3x3(x)
        obo = self.conv5x5(sbs)
        all_t = torch.cat((sbs, obo), 1)
        all_t_b = self.bcn1(all_t)
        out = torch.add(all_t_b, res)
        out = nn.ReLU(inplace=True)(out)
        out = self.bcn2(out)
        return out

class ResPath_1(nn.Module):
    def __init__(self,ch):
        super(ResPath_1,self).__init__()

        self.block = nn.Conv2d(ch,ch,3,1,1)
        self.bn=nn.BatchNorm2d(ch)
    def forward(self, x):
        x1=self.block(x)
        x1=self.bn(x1)
        x2=self.block(x1)
        x2=self.bn(x2)
        out=torch.add(x1,x2)
        return out

class MR_Att_Unet_1(nn.Module):
    def __init__(self, ch_in, ch_out,psi=False):
        super(MR_Att_Unet_1,self).__init__()
        self.ch_out=ch_out
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.cconv1=conv_block(ch_in,64)
        self.mresblock1=MultiResBlock_1(64,64)
        self.att1=Attention_block(64,64,32)
        self.res_path1 = ResPath_1(64)
        self.back_mres1 = MultiResBlock_1(128, 64)
        self.mresblock2 = MultiResBlock_1(64,128)
        self.att2 = Attention_block(128, 128, 64)
        self.res_path2 = ResPath_1(128)
        self.back_mres2 = MultiResBlock_1(256, 128)
        self.Up2=up_conv(128,64)
        self.mresblock3 = MultiResBlock_1(128,256)
        self.att3 = Attention_block(256, 256, 128)
        self.res_path3 = ResPath_1(256)
        self.back_mres3 = MultiResBlock_1(512, 256)
        self.Up3=up_conv(256,128)
        self.mresblock4 = MultiResBlock_1(256,512)
        self.att4 = Attention_block(512, 512, 256)

        self.res_path4 = ResPath_1(512)
        self.back_mres4 = MultiResBlock_1(1024, 512)
        self.Up4=up_conv(512,256)
        self.mresblock5 = MultiResBlock_1(512,1024)

        self.Up5=up_conv(1024,512)
        self.out_cov=nn.Conv2d(64,1,1,1,0)
        self.psi=psi


    def forward(self, x):
        x1=  self.cconv1(x)
        x1 = self.mresblock1(x1)
        res_x1 = self.res_path1(x1)
        x2 = self.Maxpool(x1)

        x2 = self.mresblock2(x2)
        res_x2 = self.res_path2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.mresblock3(x3)
        res_x3 = self.res_path3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.mresblock4(x4)
        res_x4 = self.res_path4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.mresblock5(x5)

        d4=self.Up5(x5)
        res_x4_1,psi4=self.att4(d4,res_x4)
        d4=torch.cat((d4,res_x4_1),dim=1)
        d4=self.back_mres4(d4)

        d3=self.Up4(d4)
        res_x3_1,psi3 = self.att3(d3, res_x3)
        d3=torch.cat((d3,res_x3_1),dim=1)
        d3=self.back_mres3(d3)

        d2=self.Up3(d3)
        res_x2_1,psi2 = self.att2(d2, res_x2)
        d2=torch.cat((d2,res_x2_1),dim=1)
        d2=self.back_mres2(d2)

        d1=self.Up2(d2)
        res_x1_1,psi1 = self.att1(d1, res_x1)
        d1=torch.cat((d1,res_x1_1),dim=1)
        d1=self.back_mres1(d1)
        d1=self.out_cov(d1)
        if self.psi:
            return psi1,psi2,psi3,psi4


        return d1


