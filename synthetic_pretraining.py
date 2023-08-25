import copy
from torchvision.models import resnet50
import math
import torch.nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from PIL import Image
# from nenc import NENC
from itertools import product
import itertools
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import numpy
import torch.nn as nn
import cv2
import time
from scipy.io import loadmat
from glob import glob
import numpy as np
from numpy.random import randint
from torch.serialization import save
from modules.loss_modular import ContrastiveLoss
import pandas as pd
# from model import Resprojhead
# from resnete import ResNet, Bottleneck
import csv
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
import json
from PIL import ImageFile
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_dir = './dataset_images/KADIS/'


save_freq = 1
scores = {}


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--level', type=int,
                        default=0, help='Level')
    parser.add_argument('--device', type=int,
                        default=0, help='Device (GPU ID)')
    parser.add_argument('--k', type=int,
                        default=4, help='Negative samples')
    parser.add_argument('--bs', type=int,
                        default=16, help='Number of scenes perbatch')
    parser.add_argument('--tao', type=float,
                        default=0.1, help='Temperature parameter')
    parser.add_argument('--allden', action='store_true',
                        help='Use both denominators together in one denominator')

    parser.add_argument('--all_loss', action='store_true',
                        help='Use all negative combinations')

    parser.add_argument('--oneanchorperk', action='store_true',
                        help='Only one anchor per k distortions')
    parser.add_argument('--log', action='store_true',
                        help='Test mode for model')

    parser.add_argument('--savefolder', type=str,
                        default='./results/syn_pre/', help='Folder in which the models are saved')

    optn = parser.parse_args()
    optn.saveloc = optn.savefolder + 'level_' + str(optn.level)

    if optn.log:
        os.makedirs(optn.saveloc, exist_ok=True)
        with open(optn.saveloc + '/config.txt', 'w') as f:
            json.dump(optn.__dict__, f, indent=2)

    return optn


args = parse_option()

# torch.cuda.set_per_process_memory_fraction(0.7, 0)

k = args.k
tao = args.tao
bs = args.bs
cuda1 = torch.device('cuda:' + str(args.device))


class US_Data(Dataset):
    def __init__(self, data_dir, k, transform=None):
        filenames = os.listdir(data_dir)
        self.pathnames = [os.path.join(data_dir, f) for f in filenames]

        # create empty list to store the combinations
        # self.pathnames = list(itertools.combinations(pathnames, 2))

        self.transform = transform
        self.k = k

    def __len__(self):
        return len(self.pathnames)

    def __getitem__(self, idx):

        folder = self.pathnames[idx]
        imgs = os.listdir(folder)
        ref = [string for string in imgs if "REF" in string]
        dists = [string for string in imgs if "REF" not in string]

        distsampled = random.sample(dists, self.k-1)
        current = ref + distsampled
        curr = current[0]

        curr_img = Image.open(folder + '/' + curr)
        curr_img = self.transform(curr_img)

        first = curr_img.unsqueeze(dim=0)
        # first = first.cuda(cuda1)
        if args.level == 0:
            first_pass = first

        for q in range(1, len(current)):
            curr = current[q]
            curr_img = Image.open(folder + '/' + curr)
            curr_img = self.transform(curr_img)

            second = curr_img.unsqueeze(dim=0)
            # second = second.cuda(cuda1)

            if args.level == 0:
                second_pass = second

            first_pass = torch.cat((first_pass, second_pass))

        batch = first_pass
        # global ps
        cps = min(int(batch.size(2)/2), int(batch.size(3)/2))

        ###twopatchsampler############
        f = randint(1, 3)
        if f == 1:
            half1, half2 = torch.split(
                batch, int(batch.size(2)/2), dim=2)

        else:
            half1, half2 = torch.split(
                batch, int(batch.size(3)/2), dim=3)

        crop = transforms.RandomCrop(cps)
        p1_batch = crop(half1)
        p2_batch = crop(half2)

        return p1_batch, p2_batch


transform_train = transforms.Compose([transforms.ToTensor()])
train_dataloader = DataLoader(US_Data(data_dir, k, transform_train),
                              batch_size=bs, shuffle=True, drop_last=True, num_workers=12)

res = resnet50()
modules = list(res.children())[:-1]
model = nn.Sequential(*modules)


def createindmat(k, bs):

    liste = []
    bs = int(k)
    views = 2
    for i in range(1, bs+1):
        for j in range(1, views+1):
            temp = 'image' + str(i) + '_' + 'view' + str(j)
            liste.append(temp)
    A = np.array(liste, object)
    B = np.array(liste, object)
    B = B.T

    C = A[None, :] + '_' + B[:, None]
    C = C.T

    return C


indmat = createindmat(k, bs)

model.train()
model = model.cuda(cuda1)
opt = optim.Adam(model.parameters(), lr=1e-2)
# scheduler = MultiStepLR(opt, milestones=[420], gamma=0.1)


criterion = ContrastiveLoss(allden=args.allden, all_loss=args.all_loss,
                    oneanchorperk=args.oneanchorperk, modes=['v1ancdisv2', 'v2ancdisv1'])
eps = 1e-8
for epoch in range(5):
    epoch_loss = 0
    start_time = time.time()

    for n_count, batch in enumerate(tqdm(train_dataloader)):
        batchtime = time.time()
        opt.zero_grad()
        p1_batch, p2_batch = Variable(batch[0]), Variable(
            batch[1])
        ps = p1_batch.size(-1)

        stacked = torch.stack((p1_batch, p2_batch), dim=2)

        stacked = stacked.view(bs*2*k, 3, ps, ps)

        passsamples = stacked.cuda(cuda1)

        feats = model(passsamples)

        feats = feats.view(bs, k, 2, -1)

        interleaved = torch.flatten(feats, start_dim=1, end_dim=2)

        Ai = interleaved
        Anorm = torch.linalg.norm(Ai, dim=2)
        a_norm = torch.max(Anorm, eps * torch.ones_like(Anorm))
        A_n = Ai / a_norm.view(bs, 2*k, 1)
        mat = torch.bmm(A_n, torch.transpose(A_n, 1, 2))
        mat = mat/tao
        loss = criterion(mat, indmat)
        epoch_loss += loss.item()
        loss.backward()
        opt.step()
        batchtimed = time.time() - batchtime

    # scheduler.step()
    elapsed_time = time.time() - start_time
    print('epoch = %4d , loss = %4.4f , time = %4.2f s' %
          (epoch + 1, epoch_loss / n_count, elapsed_time))
    if (epoch + 1) % save_freq == 0:
        torch.save(model.state_dict(), os.path.join(
            args.saveloc, 'model_%03d.pth' % (epoch+1)))
