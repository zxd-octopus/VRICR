# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np
from tqdm import tqdm

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--train_epoch", type=int, default=20000)
    cfg = parser.parse_args()
    return cfg


def one_hot(indice, num_classes):
    """
    one_hot
    """
    I = torch.eye(num_classes).to(indice.device)
    T = I[indice]
    T.requires_grad = False
    return T


class GumbelSoftmax(nn.Module):
    def __init__(self, normed=False):
        super(GumbelSoftmax, self).__init__()
        self.eps = 1e-24
        self.normed =normed

    def forward(self, inp, tau=0.05):

        if self.normed:
            inp = torch.log(inp + self.eps)
        out = F.gumbel_softmax(inp, tau)
        return out


class Argmax(nn.Module):
    def __init__(self):
        super(Argmax, self).__init__()

    def forward(self, inp):
        return torch.argmax(inp, dim=-1)


class GUMBEL(nn.Module):
    def __init__(self, sample_num, hidden_size, is_train=False, gumbel_act=True):#sample_num 从中采样的总数
        super(GUMBEL, self).__init__()
        self.is_train = is_train
        self.gumbel_act = gumbel_act
        self.train_act1 = nn.Softmax(dim=-1)
        self.train_act2 = GumbelSoftmax()
        self.test_act3 = Argmax()

    def get_act(self):
        act = self.test_act3 if not self.is_train else (self.train_act2 if self.gumbel_act else self.train_act1)
        return act

    def forward(self, sample,training=None, tau=None):
        """GUMBEL forward
        Args
        ---
        sample: A tensor shaped of [B, sample_num]
        """
        if training:
            ret = self.train_act2(sample,tau)
            tmp = torch.nonzero(self.test_act3(sample))
            print("there are {} pairs of entities has relation in training".format(len(tmp)))
        else:
            ret = self.test_act3(sample)
            tmp = torch.nonzero(ret)
            print("there are {} pairs of entities has relation infered in testing".format(len(tmp)))
            ret = torch.zeros(sample.size(), device=sample.device).scatter(-1, ret.unsqueeze(-1), 1)


        return ret