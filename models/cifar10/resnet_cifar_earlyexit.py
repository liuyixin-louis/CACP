

"""Resnet for CIFAR10 with Early Exit branches

Resnet for CIFAR10, based on "Deep Residual Learning for Image Recognition".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for the 10-class Cifar-10 dataset.

@inproceedings{DBLP:conf/cvpr/HeZRS16,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {{CVPR}},
  pages     = {770--778},
  publisher = {{IEEE} Computer Society},
  year      = {2016}
}

"""
from .resnet_cifar import BasicBlock
from .resnet_cifar import ResNetCifar
import torch.nn as nn
from  utils.early_exit import EarlyExitMgr

__all__ = ['resnet20_cifar_earlyexit', 'resnet32_cifar_earlyexit', 'resnet44_cifar_earlyexit',
           'resnet56_cifar_earlyexit', 'resnet110_cifar_earlyexit', 'resnet1202_cifar_earlyexit']

NUM_CLASSES = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_exits_def():
    exits_def = [('layer1.2.relu2', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(1600, NUM_CLASSES)))]
    return exits_def



class ResNetCifarEarlyExit(ResNetCifar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ee_mgr = EarlyExitMgr()
        self.ee_mgr.attach_exits(self, get_exits_def())

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs


def resnet20_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet44_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [7, 7, 7], **kwargs)
    return model

def resnet56_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet110_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet1202_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [200, 200, 200], **kwargs)
    return model