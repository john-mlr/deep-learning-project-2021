import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime

from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import levels_from_labelbatch, label_to_levels
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import proba_to_label


class BRAAK_attention(nn.Module):
    def __init__(self, gate=True, L=2048, size="big", dropout=False, n_classes=5):
        super(BRAAK_attention, self).__init__()
        
        self.size_dict = {"small": [L, 512, 256], "big": [L, 512, 384]}
        size = self.size_dict[size]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

        self.attention_a = [nn.Linear(size[1], size[2]), nn.Tanh()]            
        self.attention_b = [nn.Linear(size[1], size[2]), nn.Sigmoid()]            
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(size[2], 1)

        self.classifiers = CoralLayer(size_in=size[1], num_classes=n_classes)


        self.n_classes = n_classes

    def forward(self, x, label=None, instance_eval=False, return_features=False, attention_only=False):
        h = self.fc(x)
        a = self.attention_a(h)
        b = self.attention_b(h)
        A = a.mul(b)
        A = self.attention_c(A)
        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_prob = torch.sigmoid(logits)
        Y_hat  = proba_to_label(Y_prob)

        return logits, Y_prob, Y_hat, A_raw


class BRAAK_no_attention(nn.Module):
    def __init__(self, gate=True, L=2048, size="big", dropout=False, n_classes=5):
        super(BRAAK_no_attention, self).__init__()
        
        self.size_dict = {"small": [L, 512, 256], "big": [L, 512, 384]}
        size = self.size_dict[size]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

        self.classifiers = CoralLayer(size_in=size[1], num_classes=n_classes)


        self.n_classes = n_classes

    def forward(self, x, label=None, instance_eval=False, return_features=False, attention_only=False):
        h = self.fc(x)

        M = torch.mean(h, dim=0)
        logits = torch.unsqueeze(self.classifiers(M), dim=0)
        Y_prob = torch.sigmoid(logits)
        Y_hat  = proba_to_label(Y_prob)

        return logits, Y_prob, Y_hat