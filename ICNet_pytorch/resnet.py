import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision import models

class interp(nn.Module):
    def __init__(self,size=None,scale_factor=None,mode='bilinear'):
        super(interp, self).__init__()
        assert isinstance(size,int) or (isinstance(size,tuple) and len(size) == 2) or isinstance(scale_factor,float)
        self.size=size
        self.mode=mode
        self.factor=scale_factor
    def forward(self,in_tensor):
        if self.size!=None:
            return nn.functional.interpolate(in_tensor,size=self.size,mode=self.mode,align_corners=True)
        elif self.factor!=None:
            return nn.functional.interpolate(in_tensor,scale_factor=self.factor,mode=self.mode,align_corners=True)
        else:
            raise ValueError('either size or scale_factor should be defined')

class PSPDec(nn.Module):

    def __init__(self, kernel_size=None, stride=None, upsize=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride),
            interp(upsize)
        )

    def forward(self, x):
        return self.features(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_size, out_size, stride=1, padding=1,strd=1,dilation=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1,stride=strd, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size,momentum=.95)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride,
                               padding=padding,dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size,momentum=.95)
        self.conv3 = nn.Conv2d(out_size, out_size * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size * self.expansion,momentum=.95)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ICNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ICNet, self).__init__()
        #1/2
        self.interp1=interp(scale_factor=0.5)
        #1/4
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32,momentum=.95)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(32,momentum=.95)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(64,momentum=.95) 
        #1/8
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        #1/16
        self.conv3_1=self._make_layer(block, 64, 1,strd=2)
        #1/32
        self.conv3_1_sub4=interp(scale_factor=0.5)
        self.layer2 = self._make_layer(block, 64, layers[1]-1)
        self.layer3 = self._make_layer(block, 128, layers[2],padding=2,dilation=2)
        self.layer4 = self._make_layer(block, 256, layers[3],padding=4,dilation=4)
        
        self.layer5a = PSPDec((11,20), (11,20),(11,20))
        self.layer5b = PSPDec((6,10), (5,10),(11,20))
        self.layer5c = PSPDec((3,6), (4,7),(11,20))
        self.layer5d = PSPDec((1,5), (2,3),(11,20))
        self.conv_5_4_interp = nn.Sequential(
            nn.Conv2d(1024, 256, 1, padding=0, bias=False),
            nn.BatchNorm2d(256, momentum=.95),
            nn.ReLU(inplace=True),
            interp(size=(23,40))
            )
        self.sub4=nn.Conv2d(256, num_classes, 1, bias=False)
        self.conv_sub4=nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=2,dilation=2, bias=False),
            nn.BatchNorm2d(128, momentum=.95))
        self.sub2_proj=nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=2,dilation=2, bias=False),
            nn.BatchNorm2d(128, momentum=.95))
        self.sub24_sum_interp=interp(size=(45,80))
        self.sub24=nn.Conv2d(128, num_classes, 1, bias=False)
        self.conv_sub2=nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2,dilation=2, bias=False),
            nn.BatchNorm2d(128, momentum=.95))
        self.sub1=nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1,stride=2, bias=False),
            nn.BatchNorm2d(32, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1,stride=2, bias=False),
            nn.BatchNorm2d(32, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1,stride=2, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128, momentum=.95)
            )
        self.sub12_sum=interp(size=(90,160))
        self.sub124=nn.Sequential(
            nn.Conv2d(128, num_classes, 1, bias=False),
            interp(size=(360,640)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks,strd=1, stride=1,padding=1,dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=strd, bias=False),
                nn.BatchNorm2d(planes * block.expansion,momentum=.95),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,padding,strd,dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,stride,padding,strd,dilation))
        return nn.Sequential(*layers)

    def forward(self, x):

        x1 = self.interp1(x)

        x2 = self.conv1(x1)

        x3 = self.bn1(x2)
        x4 = self.relu(x3)

        x5 = self.conv2(x4)
        x6 = self.bn2(x5)
        x7 = self.relu(x6)
        
        x8 = self.conv3(x7)
        x9 = self.bn3(x8)
        x10 = self.relu(x9)

        x11 = self.maxpool(x10)

        x12 = self.layer1(x11)

        conv3_1=self.conv3_1(x12)

        conv3_1_sub4=self.conv3_1_sub4(conv3_1)

        x13 = self.layer2(conv3_1_sub4)
        x14 = self.layer3(x13)
        x15 = self.layer4(x14) 
        la1=self.layer5a(x15)
        la2=self.layer5b(x15)
        la3=self.layer5c(x15)
        la4=self.layer5d(x15)
        conv_5_4_interp=self.conv_5_4_interp(x15+la1+la2                                     +la3+la4) 
        sub4=self.sub4(conv_5_4_interp)
        conv_sub4=self.conv_sub4(conv_5_4_interp)
        sub2_proj=self.sub2_proj(conv3_1)
        sub24_sum_interp=self.sub24_sum_interp(conv_sub4+sub2_proj)
        sub24=self.sub24(sub24_sum_interp)
        conv_sub2=self.conv_sub2(sub24_sum_interp)
        sub1=self.sub1(x)
        sub12_sum=self.sub12_sum(sub1+conv_sub2)
        sub124=self.sub124(sub12_sum)
        return sub4,sub24,sub124
