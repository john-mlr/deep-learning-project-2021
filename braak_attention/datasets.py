import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file

from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import levels_from_labelbatch, label_to_levels
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import proba_to_label

import time


class EmbeddingsDataset(torch.utils.data.Dataset):
    """ Dataset for .npy saved embeddings, given a csv of cases to include.
        Will loop through each cases to a corresponding
        list of paths. Those paths are then used for getitem.
    """
    def __init__(self, csv, embed_dir):
        self.df = pd.read_csv(csv)
        self.embed_dir = embed_dir
        
        files = []
        braak = []
        for _,i in self.df.iterrows():
            file = os.path.join(embed_dir, str(int(i['ID'])) + '.npy')
            if os.path.exists(file):
                files.append(file)
                braak.append(int(i['BRAAK']))
        
        self.files = files
        self.braak = braak
        self.n_classes = len(set(braak))
        
        with open(files[0],'rb') as f:
            feat = np.load(f)
        self.feats_size = feat.shape[1]
            
                
    def __len__(self):
        return(len(self.files))
    
    def __getitem__(self, idx):
        
        with open(self.files[idx],'rb') as f:
            feats = np.load(f)
        feats = torch.from_numpy(feats)
            
        score = torch.tensor(self.braak[idx], dtype=torch.int)
        
        level = label_to_levels(score, num_classes=self.n_classes)
            
        return feats, score, level