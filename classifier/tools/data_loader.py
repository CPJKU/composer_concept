""" Slightly adapted from original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/tools/data_loader.py) """
# MIDIDataset

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from glob import glob
from classifier.tools.transformation import (
    ToTensor,
    Transpose,
    Segmentation,
    TempoStretch,
    DoubleTempo,
)
import random
import os
from config import data_root


class MIDIDataset(Dataset):
    def __init__(
        self,
        train=True,
        txt_file="",
        classes=13,
        omit=None,
        seg_num=20,
        age=False,
        transform=None,
        transpose_rng=6,
    ):
        self.is_train = train
        self.txt_file = txt_file
        self.classes = [x for x in range(classes)]

        self.seg_num = seg_num  # seg num per song
        self.transform = transform
        self.transpose_rng = transpose_rng

        self.x_path = []
        self.y = []
        self.order = []

        self.map = {}

        self.omitlist = []
        if omit:
            self.omitlist = omit.split(",")  # ['2', '5']. str list.

        # omit = list of string
        if self.omitlist is not None:
            for c in self.classes:
                if str(c) in self.omitlist:
                    continue
                label = c - sum(c > int(o) for o in self.omitlist)
                self.map[c] = label

        txt_list = open(self.txt_file, "r")
        for midi_pth in txt_list:  # each midi
            midi_pth = os.path.join(data_root, midi_pth)
            midi_pth = midi_pth.replace("\n", "")
            # print("midi_pth", midi_pth)
            path_parts = midi_pth.split("/")
            # print("path_parts", path_parts)
            comp_num = -1
            for part in path_parts:
                if "composer" in part and part != 'concept_composers':
                    comp_num = int(part.replace("composer", ""))
                    break

            ver_npy = glob(midi_pth + "/*.npy")  # list
            # randomly select n segments pth
            tmp = [random.choice(ver_npy) for j in range(self.seg_num)]
            self.x_path.extend(tmp)
            self.order.extend([k for k in range(self.seg_num)])  # seg 위치/순서
            self.y.extend([self.map[comp_num]] * self.seg_num)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = np.load(self.x_path[idx], allow_pickle=True)
        Y = self.y[idx]

        fd = self.x_path[idx].find("composer")
        pth = self.x_path[idx][fd:]

        data = {"X": X, "Y": Y, "pth": pth}

        # torch.transforms
        trans = [Segmentation(self.is_train, self.seg_num, self.order[idx])]
        if self.transform == "Transpose":
            trans.append(Transpose(self.transpose_rng))
        elif self.transform == "Tempo":
            trans.append(TempoStretch())
        elif self.transform == "DoubleTempo":
            trans.append(DoubleTempo())
        trans.append(ToTensor())
        data = transforms.Compose(trans)(data)

        return data
