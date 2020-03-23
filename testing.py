# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 04:13:36 2020

@author: Kamaljeet
"""

in_fea = 3
out1 = 16
out2 = 32
out_fea = 2

pad = nn.ZeroPad2d(2)
conv1 = nn.Conv2d(in_fea, out1, 5)
pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
conv2 = nn.Conv2d(out1, out2, 5)
fc1 = nn.Linear(out2*300*300, 128)
fc2 = nn.Linear(128,64)
fc3 = nn.Linear(64, out_fea)

a1 = pad(images)
a2 = conv1(a1)
a3 = pool(F.relu(a2))

a4 = pool(F.relu(conv2(pad(a3))))
