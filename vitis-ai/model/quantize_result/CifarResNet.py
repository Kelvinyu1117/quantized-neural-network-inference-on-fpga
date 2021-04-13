# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class CifarResNet(torch.nn.Module):
    def __init__(self):
        super(CifarResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #CifarResNet::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Conv2d[conv1]/input.2
        self.module_3 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/input.3
        self.module_4 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]/input.4
        self.module_6 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[0]/input.5
        self.module_7 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]/input.6
        self.module_9 = py_nndct.nn.Add() #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[0]/out.2
        self.module_10 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[0]/input.7
        self.module_11 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]/input.8
        self.module_13 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[1]/input.9
        self.module_14 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/input.10
        self.module_16 = py_nndct.nn.Add() #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[1]/out.3
        self.module_17 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[1]/input.11
        self.module_18 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[2]/Conv2d[conv1]/input.12
        self.module_20 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[2]/input.13
        self.module_21 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[2]/Conv2d[conv2]/input.14
        self.module_23 = py_nndct.nn.Add() #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[2]/out.4
        self.module_24 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer1]/BasicBlock[2]/input.15
        self.module_25 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]/input.16
        self.module_27 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[0]/input.17
        self.module_28 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]/input.18
        self.module_30 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[0]/Sequential[shortcut]/Conv2d[0]/input.19
        self.module_32 = py_nndct.nn.Add() #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[0]/out.5
        self.module_33 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[0]/input.20
        self.module_34 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]/input.21
        self.module_36 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[1]/input.22
        self.module_37 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]/input.23
        self.module_39 = py_nndct.nn.Add() #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[1]/out.6
        self.module_40 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[1]/input.24
        self.module_41 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[2]/Conv2d[conv1]/input.25
        self.module_43 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[2]/input.26
        self.module_44 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[2]/Conv2d[conv2]/input.27
        self.module_46 = py_nndct.nn.Add() #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[2]/out.7
        self.module_47 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer2]/BasicBlock[2]/input.28
        self.module_48 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]/input.29
        self.module_50 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[0]/input.30
        self.module_51 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]/input.31
        self.module_53 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[0]/Sequential[shortcut]/Conv2d[0]/input.32
        self.module_55 = py_nndct.nn.Add() #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[0]/out.8
        self.module_56 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[0]/input.33
        self.module_57 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]/input.34
        self.module_59 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[1]/input.35
        self.module_60 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]/input.36
        self.module_62 = py_nndct.nn.Add() #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[1]/out.9
        self.module_63 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[1]/input.37
        self.module_64 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[2]/Conv2d[conv1]/input.38
        self.module_66 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[2]/input.39
        self.module_67 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[2]/Conv2d[conv2]/input.40
        self.module_69 = py_nndct.nn.Add() #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[2]/out
        self.module_70 = py_nndct.nn.ReLU(inplace=False) #CifarResNet::CifarResNet/Sequential[layer3]/BasicBlock[2]/501
        self.module_71 = py_nndct.nn.AvgPool2d(kernel_size=[8, 8], stride=[8, 8], padding=[0, 0], ceil_mode=False, count_include_pad=True) #CifarResNet::CifarResNet/508
        self.module_72 = py_nndct.nn.Module('shape') #CifarResNet::CifarResNet/510
        self.module_73 = py_nndct.nn.Module('reshape') #CifarResNet::CifarResNet/input
        self.module_74 = py_nndct.nn.Linear(in_features=64, out_features=10, bias=True) #CifarResNet::CifarResNet/Linear[linear]/517

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_3 = self.module_3(self.output_module_1)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_6 = self.module_6(self.output_module_4)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_9 = self.module_9(other=self.output_module_3, alpha=1, input=self.output_module_7)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_13 = self.module_13(self.output_module_11)
        self.output_module_14 = self.module_14(self.output_module_13)
        self.output_module_16 = self.module_16(other=self.output_module_10, alpha=1, input=self.output_module_14)
        self.output_module_17 = self.module_17(self.output_module_16)
        self.output_module_18 = self.module_18(self.output_module_17)
        self.output_module_20 = self.module_20(self.output_module_18)
        self.output_module_21 = self.module_21(self.output_module_20)
        self.output_module_23 = self.module_23(other=self.output_module_17, alpha=1, input=self.output_module_21)
        self.output_module_24 = self.module_24(self.output_module_23)
        self.output_module_25 = self.module_25(self.output_module_24)
        self.output_module_27 = self.module_27(self.output_module_25)
        self.output_module_28 = self.module_28(self.output_module_27)
        self.output_module_30 = self.module_30(self.output_module_24)
        self.output_module_32 = self.module_32(other=self.output_module_30, alpha=1, input=self.output_module_28)
        self.output_module_33 = self.module_33(self.output_module_32)
        self.output_module_34 = self.module_34(self.output_module_33)
        self.output_module_36 = self.module_36(self.output_module_34)
        self.output_module_37 = self.module_37(self.output_module_36)
        self.output_module_39 = self.module_39(other=self.output_module_33, alpha=1, input=self.output_module_37)
        self.output_module_40 = self.module_40(self.output_module_39)
        self.output_module_41 = self.module_41(self.output_module_40)
        self.output_module_43 = self.module_43(self.output_module_41)
        self.output_module_44 = self.module_44(self.output_module_43)
        self.output_module_46 = self.module_46(other=self.output_module_40, alpha=1, input=self.output_module_44)
        self.output_module_47 = self.module_47(self.output_module_46)
        self.output_module_48 = self.module_48(self.output_module_47)
        self.output_module_50 = self.module_50(self.output_module_48)
        self.output_module_51 = self.module_51(self.output_module_50)
        self.output_module_53 = self.module_53(self.output_module_47)
        self.output_module_55 = self.module_55(other=self.output_module_53, alpha=1, input=self.output_module_51)
        self.output_module_56 = self.module_56(self.output_module_55)
        self.output_module_57 = self.module_57(self.output_module_56)
        self.output_module_59 = self.module_59(self.output_module_57)
        self.output_module_60 = self.module_60(self.output_module_59)
        self.output_module_62 = self.module_62(other=self.output_module_56, alpha=1, input=self.output_module_60)
        self.output_module_63 = self.module_63(self.output_module_62)
        self.output_module_64 = self.module_64(self.output_module_63)
        self.output_module_66 = self.module_66(self.output_module_64)
        self.output_module_67 = self.module_67(self.output_module_66)
        self.output_module_69 = self.module_69(other=self.output_module_63, alpha=1, input=self.output_module_67)
        self.output_module_70 = self.module_70(self.output_module_69)
        self.output_module_71 = self.module_71(self.output_module_70)
        self.output_module_72 = self.module_72(input=self.output_module_71, dim=0)
        self.output_module_73 = self.module_73(input=self.output_module_71, size=[self.output_module_72,-1])
        self.output_module_74 = self.module_74(self.output_module_73)
        return self.output_module_74
