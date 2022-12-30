import argparse
import os
from os.path import join

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from genericpath import isdir
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

from datasets.test_datasets import CID2013, LIVE_FB, KONIQ_10k, LIVE_Challenge
from modules.niqe_compute import NIQE


def cov(tensor, rowvar=False, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--model_weights', type=str,
                        default='./pre_trained_models/003_auth_ft_cd_4.pth', help='Saved weights')
    parser.add_argument('--eval_result_dir', type=str,
                        default='./results/auth_ft_cd/', help='The directory in which the model needs to be saved')

    parser.add_argument('--patch_size', default=96, type=int,
                        help='Patch size for pristine patches')
    parser.add_argument('--sharpness_param', default=0.75, type=float,
                        help='Sharpness parameter for selecting pristine patches')
    parser.add_argument('--colorfulness_param', default=0.8, type=float,
                        help='Colorfulness parameter for selecting pristine patches')

    parser.add_argument('--dataset', type=str,
                        default='LIVEC', help='Test dataset (LIVEC/KONIQ/LIVEFB/CID)')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='Batch size')
    parser.add_argument('--device', type=str,
                        default='cuda:0', help='Device (cpu/cuda)')


    optn = parser.parse_args()

    return optn


def gaussian_filter(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size).to(dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


def select_patches(all_patches):
    selected_patches = torch.empty(1, all_patches.size(
        1), all_patches.size(2), all_patches.size(3))
    selected_patches = selected_patches.to(args.device)

    kernel_size = 7
    kernel_sigma = float(7 / 6)
    deltas = []
    for ix in range(all_patches.size(0)):
        rest = all_patches[ix, :, :, :]
        rest = rest.unsqueeze(dim=0)
        rest = transforms.Grayscale()(rest)
        kernel = gaussian_filter(kernel_size=kernel_size, sigma=kernel_sigma).view(
            1, 1, kernel_size, kernel_size).to(rest)
        C = 1
        mu = F.conv2d(rest, kernel, padding=kernel_size // 2)
        mu_sq = mu ** 2
        std = F.conv2d(rest ** 2, kernel, padding=kernel_size // 2)
        std = ((std - mu_sq).abs().sqrt())
        delta = torch.sum(std)
        deltas.append([delta])
    peak_sharpness = max(deltas)[0].item()
    for ix in range(all_patches.size(0)):
        tempdelta = deltas[ix][0].item()
        if tempdelta > p*peak_sharpness:
            selected_patches = torch.cat(
                (selected_patches, all_patches[ix, :, :, :].unsqueeze(dim=0)))
    selected_patches = selected_patches[1:, :, :, :]
    return selected_patches


def select_colorful_patches(all_patches):
    selected_patches = torch.empty(1, all_patches.size(
        1), all_patches.size(2), all_patches.size(3))
    selected_patches = selected_patches.to(args.device)
    deltas = []
    for ix in range(all_patches.size(0)):
        rest = all_patches[ix, :, :, :]
        R = rest[0, :, :]
        G = rest[1, :, :]
        B = rest[2, :, :]
        rg = torch.abs(R - G)
        yb = torch.abs(0.5 * (R + G) - B)
        rbMean = torch.mean(rg)
        rbStd = torch.std(rg)
        ybMean = torch.mean(yb)
        ybStd = torch.std(yb)
        stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))

        delta = stdRoot + meanRoot
        deltas.append([delta])
    peak_sharpness = max(deltas)[0].item()
    for ix in range(all_patches.size(0)):
        tempdelta = deltas[ix][0].item()
        if tempdelta > pc*peak_sharpness:
            selected_patches = torch.cat(
                (selected_patches, all_patches[ix, :, :, :].unsqueeze(dim=0)))
    selected_patches = selected_patches[1:, :, :, :]
    return selected_patches


with torch.no_grad():
    args = parse_option()
    if not os.path.isdir(args.eval_result_dir):
        os.makedirs(args.eval_result_dir)
    toten = transforms.ToTensor()
    refs = os.listdir('./dataset_images/pristine/')
    ps = args.patch_size
    p = args.sharpness_param
    pc = args.colorfulness_param
    if not os.path.isfile('./dataset_images/pristine_patches_%03d_%0.2f_%0.2f.hdf5' % (args.patch_size, args.sharpness_param, args.colorfulness_param)):
        temp = np.array(Image.open('./dataset_images/pristine/' + refs[0]))
        toten = transforms.ToTensor()
        temp = toten(temp)
        batch = temp.to(args.device)
        batch = batch.unsqueeze(dim=0)
        patches = batch.unfold(1, 3, 3).unfold(
            2, ps, ps).unfold(3, ps, ps)

        patches = patches.contiguous().view(1, -1, 3,
                                            ps, ps)

        for ix in range(patches.size(0)):
            patches[ix, :, :, :, :] = patches[ix, torch.randperm(
                patches.size()[1]), :, :, :]
        first_patches = patches.squeeze()
        first_patches = select_colorful_patches(select_patches(first_patches))
        # first_patches = select_colorful_patches(first_patches)


        refs = refs[1:]
        for irx, rs in enumerate(tqdm(refs)):
            temp = np.array(Image.open('./dataset_images/pristine/' + rs))
            toten = transforms.ToTensor()
            temp = toten(temp)
            batch = temp.to(args.device)
            batch = batch.unsqueeze(dim=0)
            patches = batch.unfold(1, 3, 3).unfold(
                2, ps, ps).unfold(3, ps, ps)

            patches = patches.contiguous().view(1, -1, 3,
                                                ps, ps)

            for ix in range(patches.size(0)):
                patches[ix, :, :, :, :] = patches[ix, torch.randperm(
                    patches.size()[1]), :, :, :]
            second_patches = patches.squeeze()
            second_patches = select_colorful_patches(select_patches(second_patches))
            first_patches = torch.cat((first_patches, second_patches))
        
        with h5py.File('./dataset_images/pristine_patches_%03d_%0.2f_%0.2f.hdf5' % (args.patch_size, args.sharpness_param, args.colorfulness_param), 'w') as f:
            dset = f.create_dataset('data', data = np.array(first_patches.detach().cpu(), dtype=np.float32))
    else:
        print('Using pre-selected pristine patches')
        with h5py.File('./dataset_images/pristine_patches_%03d_%0.2f_%0.2f.hdf5' % (args.patch_size, args.sharpness_param, args.colorfulness_param), 'r') as f:
            first_patches = torch.tensor(f['data'][:], device=args.device)
    
    perf = []

    res = resnet50()
    modules = list(res.children())[:-1]
    model = nn.Sequential(*modules)
    model.load_state_dict(torch.load(args.model_weights, map_location='cpu'))
    model = model.to(args.device)
    model.eval()

    all_ref_feats = (model(first_patches)).squeeze()

    niqe_model = NIQE(all_ref_feats).to(args.device)

    print("computed vr and sigmar")

    ## Change the image locations of authentically distorted images accordingly by changing img_dir.

    if args.dataset == 'LIVEC':
        img_dir = './dataset_images/LIVEC/Images/'
        data_loc = './datasets/LIVEC.csv'
        dataset = LIVE_Challenge(img_dir, data_loc)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'KONIQ':
        img_dir = './dataset_images/KONIQ/1024x768/'
        data_loc = './datasets/KONIQ.csv'
        dataset = KONIQ_10k(img_dir, data_loc)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'LIVEFB':
        img_dir = './dataset_images/LIVEFB/'
        data_loc = './datasets/LIVEFB.csv'
        dataset = LIVE_FB(img_dir, data_loc)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'CID':
        img_dir = './dataset_images/CID2013/'
        data_loc = './datasets/CID2013.csv'
        dataset = CID2013(img_dir, data_loc)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


    scores = []
    moss = []
    names = []
    for batch, (x, y, name) in enumerate(tqdm(loader)):

        x = x.to(args.device)

        x = x.unfold(-3, x.size(-3), x.size(-3)).unfold(-3, ps, int(ps/2)).unfold(-3, ps, int(ps/2)).squeeze(1)
        x = x.contiguous().view(x.size(0), x.size(1)*x.size(2), x.size(3), x.size(4), x.size(5))
        patches = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        
        all_rest_feats = (model(patches))
        
        all_rest_feats = all_rest_feats.view(x.size(0), x.size(1), -1)

        score = niqe_model(all_rest_feats)

        scores.extend(score.cpu().detach().tolist())
        moss.extend(y.tolist())
        names.extend(list(name))
    
    df_scores = pd.DataFrame()
    df_scores['file_name'] = names
    df_scores['mos'] = moss
    df_scores['score'] = scores
    df_scores.to_csv(join(args.eval_result_dir, f'%s_predictions.csv' % args.dataset))

    rho_test, p_test = spearmanr(np.array(moss), np.array(scores))
    print(rho_test)
    perf.append([args.dataset, rho_test])
    res = pd.DataFrame(perf)
    res.to_csv(join(args.eval_result_dir, f'%s_perf.csv' % args.dataset))
