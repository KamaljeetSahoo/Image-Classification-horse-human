# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 03:01:59 2020

@author: Kamaljeet
"""

import torch
import torchvision
import torchvision.transforms as transforms



def gwt_data(train_folder, test_folder):
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.ImageFolder(root = train_folder, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.ImageFolder(root = test_folder, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    
    return trainloader, testloader, trainset.classes