
# The TorchVision implementation in https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# has 2 issues in the implementation of the BasicBlock and Bottleneck modules, which impact our ability to
# collect activation statistics and run quantization:
#   1. Re-used ReLU modules
#   2. Element-wise addition as a direct tensor operation
# Here we provide an implementation of both classes that fixes these issues, and we provide the same API to create
# ResNet and ResNeXt models as in the TorchVision implementation.
# We reuse the original implementation as much as possible.

from collections import OrderedDict
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, _resnet

from modules import EltwiseAdd


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2',
           'Bottleneck']


class BasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        # Initialize torchvision version
        super(BasicBlock, self).__init__(*args, **kwargs)

        # Remove original relu in favor of numbered modules
        delattr(self, 'relu')
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.add = EltwiseAdd(inplace=True)  # Replace '+=' operator with inplace module

        # Trick to make the modules accessible in their topological order
        modules = OrderedDict()
        modules['conv1'] = self.conv1
        modules['bn1'] = self.bn1
        modules['relu1'] = self.relu1
        modules['conv2'] = self.conv2
        modules['bn2'] = self.bn2
        if self.downsample is not None:
            modules['downsample'] = self.downsample
        modules['add'] = self.add
        modules['relu2'] = self.relu2
        self._modules = modules

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu2(out)

        return out


class Bottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        # Initialize torchvision version
        super(Bottleneck, self).__init__(*args, **kwargs)

        # Remove original relu in favor of numbered modules
        delattr(self, 'relu')
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.add = EltwiseAdd(inplace=True)  # Replace '+=' operator with inplace module

        # Trick to make the modules accessible in their topological order
        modules = OrderedDict()
        modules['conv1'] = self.conv1
        modules['bn1'] = self.bn1
        modules['relu1'] = self.relu1
        modules['conv2'] = self.conv2
        modules['bn2'] = self.bn2
        modules['relu2'] = self.relu2
        modules['conv3'] = self.conv3
        modules['bn3'] = self.bn3
        if self.downsample is not None:
            modules['downsample'] = self.downsample
        modules['add'] = self.add
        modules['relu3'] = self.relu3
        self._modules = modules

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu3(out)

        return out


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    """Constructs a Wide ResNet-50-2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    """Constructs a Wide ResNet-101-2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
