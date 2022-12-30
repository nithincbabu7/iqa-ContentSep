import random
from numpy.random import randint
from scipy.io.matlab.miobase import matdims
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
import time
from itertools import chain, combinations
import torch.nn.functional as f


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


allmodes = ['v1ancdisv1', 'v1ancdisv2', 'v2ancdisv1', 'v2ancdisv2']

allco = powerset(allmodes)


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, modes, allden, all_loss, oneanchorperk):
        super(ContrastiveLoss, self).__init__()
        self.modes = modes
        self.allden = allden
        self.all_loss = all_loss
        self.oneanchorperk = oneanchorperk

    def forward(self, mat, indmat):
        times = time.time()
        loss_sum = 0

        ret = []
        modes = self.modes

        v1ancdisv1_ancs = -99*torch.ones(mat.size(0), int(mat.size(1)/2), 1)
        v1ancdisv2_ancs = -99*torch.ones(mat.size(0), int(mat.size(1)/2), 1)
        v2ancdisv1_ancs = -99*torch.ones(mat.size(0), int(mat.size(1)/2), 1)
        v2ancdisv2_ancs = -99*torch.ones(mat.size(0), int(mat.size(1)/2), 1)

        v1ancdisv1_negs = -99 * \
            torch.ones(mat.size(0), int(mat.size(1)/2),
                       int((mat.size(1) - 2)/2))
        v1ancdisv2_negs = -99 * \
            torch.ones(mat.size(0), int(mat.size(1)/2),
                       int((mat.size(1) - 2)/2))
        v2ancdisv1_negs = -99 * \
            torch.ones(mat.size(0), int(mat.size(1)/2),
                       int((mat.size(1) - 2)/2))
        v2ancdisv2_negs = -99 * \
            torch.ones(mat.size(0), int(mat.size(1)/2),
                       int((mat.size(1) - 2)/2))

        for i in range(0, int((np.shape(mat)[1]))):
            if i % 2 == 0:
                anc_ind = i+1
            else:
                anc_ind = i-1

            temprow = mat[:, i, :]

            indrow = list(indmat[i, :])
            indrow = [i.split('_') for i in indrow]

            anc = [indrow[anc_ind][0], indrow[anc_ind][1]]

            if i % 2 == 0:

                v1ancdisv1_negs_l = [s for s in indrow if s[3] == anc[1]]
                v1ancdisv2_negs_l = [s for s in indrow if s[3] != anc[1]]

                v1ancdisv1_negs_l = [
                    s for s in v1ancdisv1_negs_l if s[0] != s[2]]
                v1ancdisv2_negs_l = [
                    s for s in v1ancdisv2_negs_l if s[0] != s[2]]

                v1ancdisv1_negs_ind = [i for i, e in enumerate(
                    indrow) if e in v1ancdisv1_negs_l]
                v1ancdisv2_negs_ind = [i for i, e in enumerate(
                    indrow) if e in v1ancdisv2_negs_l]
                v1ancdisv1_negs_list = temprow[:, v1ancdisv1_negs_ind]
                v1ancdisv2_negs_list = temprow[:, v1ancdisv2_negs_ind]
                num = temprow[:, anc_ind].unsqueeze(dim=1)
                v1ancdisv1_ancs[:, math.floor(i/2), :] = num
                v1ancdisv2_ancs[:, math.floor(i/2), :] = num
                v1ancdisv1_negs[:, math.floor(i/2), :] = v1ancdisv1_negs_list
                v1ancdisv2_negs[:, math.floor(i/2), :] = v1ancdisv2_negs_list

            else:

                v2ancdisv2_negs_l = [s for s in indrow if s[3] == anc[1]]
                v2ancdisv1_negs_l = [s for s in indrow if s[3] != anc[1]]

                v2ancdisv2_negs_l = [
                    s for s in v2ancdisv2_negs_l if s[0] != s[2]]
                v2ancdisv1_negs_l = [
                    s for s in v2ancdisv1_negs_l if s[0] != s[2]]

                v2ancdisv2_negs_ind = [i for i, e in enumerate(
                    indrow) if e in v2ancdisv2_negs_l]
                v2ancdisv1_negs_ind = [i for i, e in enumerate(
                    indrow) if e in v2ancdisv1_negs_l]
                v2ancdisv2_negs_list = temprow[:, v2ancdisv2_negs_ind]
                v2ancdisv1_negs_list = temprow[:, v2ancdisv1_negs_ind]
                num = temprow[:, anc_ind].unsqueeze(dim=1)
                v2ancdisv2_ancs[:, math.floor(i/2), :] = num
                v2ancdisv1_ancs[:, math.floor(i/2), :] = num
                v2ancdisv2_negs[:, math.floor(i/2), :] = v2ancdisv2_negs_list
                v2ancdisv1_negs[:, math.floor(i/2), :] = v2ancdisv1_negs_list

        v1ancdisv1_set = torch.cat(
            (v1ancdisv1_ancs, v1ancdisv1_negs), dim=2).unsqueeze(dim=0)
        v1ancdisv2_set = torch.cat(
            (v1ancdisv2_ancs, v1ancdisv2_negs), dim=2).unsqueeze(dim=0)
        v2ancdisv1_set = torch.cat(
            (v2ancdisv1_ancs, v2ancdisv1_negs), dim=2).unsqueeze(dim=0)
        v2ancdisv2_set = torch.cat(
            (v2ancdisv2_ancs, v2ancdisv2_negs), dim=2).unsqueeze(dim=0)

        # allmodes = ['v1ancdisv1','v1ancdisv2','v2ancdisv1','v2ancdisv2']
        allmodesdict = {'v1ancdisv1': v1ancdisv1_set, 'v1ancdisv2': v1ancdisv2_set,
                        'v2ancdisv1': v2ancdisv1_set, 'v2ancdisv2': v2ancdisv2_set}

        # allco = powerset(allmodes)
        loss_set = torch.zeros(len(self.modes), v1ancdisv1_set.size(
            1), v1ancdisv1_set.size(2), v1ancdisv1_set.size(3))
        # for cont,subset in enumerate(allco):
        #     if len(list(subset)):
        #         if list(subset)==modes:
        for ix, x in enumerate(modes):
            loss_set[ix, :, :, :] = allmodesdict[x]

        loss_set = torch.exp(loss_set)
        if self.allden:
            red_loss_set = torch.sum(loss_set, dim=0)
            red_loss_set[:, :, 0] = red_loss_set[:, :, 0]/len(self.modes)
            red_loss_set = red_loss_set.unsqueeze(dim=0)

        else:
            red_loss_set = loss_set
        if self.oneanchorperk:
            red_loss_set = red_loss_set[:, :, 0, :]
            red_loss_set = red_loss_set.unsqueeze(dim=-2)

        out = f.normalize(red_loss_set, p=1, dim=3)

        cal = out[:, :, :, 0]

        step = -torch.log(cal)
        dist_avg = torch.mean(step, dim=2)
        batch_avg = torch.mean(dist_avg, dim=1)
        loss_sum = torch.sum(batch_avg, dim=0)

        # loss = torch.sum(-torch.log(cal))/
        timed = time.time() - times
        return loss_sum


def D(p, z):  # negative cosine similarity
    z = z.detach()  # stop gradient
    p = F.normalize(p, dim=1)  # l2-normalize

    z = F.normalize(z, dim=1)  # l2-normalize
    out = -(p*z).sum(dim=1).mean()

    return out


class BYOL_loss(torch.nn.Module):

    def __init__(self):
        super(BYOL_loss, self).__init__()

    def forward(self, z1, z2, p1, p2):
        L = D(p1, z2)/2 + D(p2, z1)/2

        return L