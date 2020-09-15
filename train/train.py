from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch
import torch.nn as nn


class EltwiseAdd(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise addition"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res

#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Resnet for CIFAR10

Resnet for CIFAR10, based on "Deep Residual Learning for Image Recognition".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for the 10-class Cifar-10 dataset.
This ResNet also has layer gates, to be able to dynamically remove layers.

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
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
# from modules import EltwiseAdd


__all__ = ['resnet20_cifar', 'resnet32_cifar', 'resnet44_cifar', 'resnet56_cifar']

NUM_CLASSES = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.residual_eltwiseadd = EltwiseAdd()

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

        if self.block_gates[1]:
            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.residual_eltwiseadd(residual, out)
        out = self.relu2(out)

        return out


class ResNetCifar(nn.Module):

    def __init__(self, block, layers, num_classes=NUM_CLASSES):
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16  # 64
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 16, layers[0])
        self.layer2 = self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_cifar(**kwargs):
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar(**kwargs):
    model = ResNetCifar(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet44_cifar(**kwargs):
    model = ResNetCifar(BasicBlock, [7, 7, 7], **kwargs)
    return model

def resnet56_cifar(**kwargs):
    model = ResNetCifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os

transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform1 = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/home/dataset/cifar', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/dataset/cifar', train=False,
                                       download=True, transform=transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = resnet56_cifar()


loss = nn.CrossEntropyLoss()
#optimizer = optim.SGD(self.parameters(),lr=0.01)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for epoch in range(100):  # loop over the dataset multiple times
    timestart = time.time()

    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device),labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()

        # print statistics
        running_loss += l.item()
        # print("i ",i)
        if i % 500 == 499:  # print every 500 mini-batches
            print('[%d, %5d] loss: %.4f' %
                    (epoch, i, running_loss / 500))
            running_loss = 0.0
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the %d tran images: %.3f %%' % (total,
                    100.0 * correct / total))
            total = 0
            correct = 0

    print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))

print('Finished Training')


# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

# def main():
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F
#     import torchvision
#     import torchvision.transforms as transforms
#     import torch.optim as optim
#     import time
#     import os

#     transform = transforms.Compose(
#         [
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomGrayscale(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     transform1 = transforms.Compose(
#         [
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
#                                             shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform1)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=50,
#                                             shuffle=False, num_workers=2)

#     classes = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#     # Training settings
#     # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#     #                     help='input batch size for training (default: 64)')
#     # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#     #                     help='input batch size for testing (default: 1000)')
#     # parser.add_argument('--epochs', type=int, default=10, metavar='N',
#     #                     help='number of epochs to train (default: 10)')
#     # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#     #                     help='learning rate (default: 0.01)')
#     # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#     #                     help='SGD momentum (default: 0.5)')
#     # parser.add_argument('--no-cuda', action='store_true', default=False,
#     #                     help='disables CUDA training')
#     # parser.add_argument('--seed', type=int, default=1, metavar='S',
#     #                     help='random seed (default: 1)')
#     # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#     #                     help='how many batches to wait before logging training status')
   
#     # parser.add_argument('--save-model', action='store_true', default=False,
#     #                     help='For Saving the current Model')
#     # args = parser.parse_args()
#     # use_cuda = not args.no_cuda and torch.cuda.is_available()

#     # torch.manual_seed(args.seed)

#     # device = torch.device("cuda" if use_cuda else "cpu")

#     # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#     # train_loader = torch.utils.data.DataLoader(
#     #     datasets.MNIST('../data', train=True, download=True,
#     #                    transform=transforms.Compose([
#     #                        transforms.ToTensor(),
#     #                        transforms.Normalize((0.1307,), (0.3081,))
#     #                    ])),
#     #     batch_size=args.batch_size, shuffle=True, **kwargs)
#     # test_loader = torch.utils.data.DataLoader(
#     #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#     #                        transforms.ToTensor(),
#     #                        transforms.Normalize((0.1307,), (0.3081,))
#     #                    ])),
#     #     batch_size=args.test_batch_size, shuffle=True, **kwargs)


#     # model = Net().to(device)
#     # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#     # for epoch in range(1, args.epochs + 1):
#     #     train(args, model, device, train_loader, optimizer, epoch)
#     #     test(args, model, device, test_loader)

#     # if (args.save_model):
#     #     torch.save(model.state_dict(),"mnist_cnn.pt")
       
# if __name__ == '__main__':
#     main()

