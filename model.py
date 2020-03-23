# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:49:26 2020

@author: Kamaljeet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, out1, out2, in_fea = 3, out_fea = 2):
        super(Classifier, self).__init__()
        self.out2 = out2
        self.pad = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(in_fea, out1, 5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(out1, out2, 5)
        self.fc1 = nn.Linear(out2*75*75, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64, out_fea)
        
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(self.pad(x))))
        x = F.relu(self.pool(self.conv2(self.pad(x))))
        x = x.view(-1, self.out2*75*75)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

