from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or issubclass(type(m), nn.Linear) or issubclass(type(m), nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride=stride)

        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes, stride=1)

        self.bn2 = nn.BatchNorm2d(out_planes)

        self.act = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * out_planes, stride),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _make_layer(block, in_planes, out_planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
        layers.append(block(in_planes=in_planes,
                            out_planes=out_planes, stride=stride))
        in_planes = out_planes * block.expansion

    return (in_planes, nn.Sequential(*layers))


class CifarResNet(Module):
    def __init__(self, block: BasicBlock, num_blocks, num_classes=10):
        super(CifarResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(in_planes=3, out_planes=16, stride=1)

        self.bn1 = nn.BatchNorm2d(16)

        self.in_planes, self.layer1 = _make_layer(
            block, in_planes=self.in_planes, out_planes=16, num_blocks=num_blocks[0], stride=1)

        self.in_planes, self.layer2 = _make_layer(
            block, in_planes=self.in_planes, out_planes=32, num_blocks=num_blocks[1], stride=2)

        self.in_planes, self.layer3 = _make_layer(
            block, in_planes=self.in_planes, out_planes=64, num_blocks=num_blocks[2], stride=2)

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = nn.functional.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if(torch.is_tensor(out.size()[3])):
            out = nn.functional.avg_pool2d(out, out.size()[3].item())
        else:
            out = nn.functional.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(no_quant=True):
    return CifarResNet(BasicBlock, [3, 3, 3], num_classes=10)
