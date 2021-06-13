import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from collections import OrderedDict

from small_cnn import * 
from resnet import * 
from wideresnet import * 

def swish(x):
    return x * F.sigmoid(x)

def activ(key): 
    if key == 'relu': 
        return nn.ReLU() 
    elif key == 'elu': 
        return nn.ELU()
    elif key == 'swish': 
        return swish

def get_model(ds, model, activation=nn.ReLU()): 
    if ds == 'toy2d': 
        return Toy2D()
    
    elif ds == 'mnist': 
        assert (model=='cnn')
        return Mnist(activation=activation)

    elif ds == 'cifar10': 
        if model == 'cnn': 
            return Cifar10(activation=activation)
        elif model == 'resnet18': 
            return ResNet18()
        elif model == 'wideresnet': 
            return WideResNet()
    
def adjust_learning_rate_mnist(optimizer, epoch, lr):
    """decrease the learning rate"""
    _lr = lr 
    if epoch >= 55:
        _lr = lr * 0.1
    if epoch >= 75:
        _lr = lr * 0.01
    if epoch >= 90:
        _lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr
    return optimizer

def adjust_learning_rate_cifar10(optimizer, epoch, lr):
    """decrease the learning rate"""
    _lr = lr
    if epoch >= 75:
        _lr = lr * 0.1
    if epoch >= 90:
        _lr = lr * 0.01
    if epoch >= 100:
        _lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr
    return optimizer

def adjust_learning_rate(optimizer, epoch, lr, ds): 
    if ds == 'mnist': 
        return adjust_learning_rate_mnist(optimizer, epoch, lr)
    elif ds in ['cifar10', 'cifar100']: 
        return adjust_learning_rate_cifar10(optimizer, epoch, lr)


def get_optimizer(ds, model, architecture):
    if ds == 'mnist': 
        assert(architecture == 'cnn')
        lr = 0.01
        momentum = 0.9
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)             

    elif ds == 'cifar10': 
        if architecture == 'cnn':
            lr = 0.001
            opt = optim.Adam(model.parameters(), lr=lr)
        
        elif architecture == 'resnet18': 
            lr = 0.01
            momentum = 0.9
            weight_decay = 3.5e-3 # MART SETTING
            opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)    

        elif architecture == 'wideresnet': 
            lr = 0.1
            momentum = 0.9
            weight_decay = 2e-4
            opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)            
    
    else: 
        raise ValueError

    return opt, lr

def switch_status(model, status): 
    if status == 'train': 
        model.train()
    elif status == 'eval': 
        model.eval()
    else: 
        raise ValueError
#------------------------------------------------------
class DataWithIndex(Dataset):
    def __init__(self, train_data):
        self.data = train_data
        
    def __getitem__(self, index):
        data, target = self.data[index]
        
        return data, target, index

    def __len__(self):
        return len(self.data)