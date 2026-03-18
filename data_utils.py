import random
import numpy as np
import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, point_clouds, labels):
        self.point_clouds = point_clouds
        self.labels = labels

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return self.point_clouds[idx], self.labels[idx]


def load_h5_ModelNet(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        data = f["data"][:]
        label = f["label"][:]
    return data, label


def getfile(path):
    return [line.strip('\n') for line in open(path)]


def load_data(dataset, batch_size):

    if dataset == 'modelnet40':

        # Replace this with your point cloud dataset
        point_clouds_train = []
        pids_train = []
        trainfiles=getfile(os.path.join('objdata/ModelNet40/train_files.txt'))
        for i in range(len(trainfiles)):
            traindata, labels = load_h5_ModelNet('objdata/ModelNet40/'+trainfiles[i])
            for j in range(len(traindata)):
                point_clouds_train.append(torch.from_numpy(traindata[j]).float())
                pids_train.append(torch.from_numpy(labels[j]).long())

        # Replace this with your point cloud dataset
        point_clouds_test = []
        pids_test = []
        testfiles=getfile(os.path.join('objdata/ModelNet40/test_files.txt'))
        for i in range(len(testfiles)):
            testdata, labels = load_h5_ModelNet('objdata/ModelNet40/'+testfiles[i])
            for j in range(len(testdata)):
                point_clouds_test.append(torch.from_numpy(testdata[j]).float())
                pids_test.append(torch.from_numpy(labels[j]).long())

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_dataset = PointCloudDataset(point_clouds_train, pids_train)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = PointCloudDataset(point_clouds_test, pids_test)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataset, dataloader_train, test_dataset, dataloader_test
