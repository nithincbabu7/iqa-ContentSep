from numpy.random import randint
from torch.serialization import save
from modules.loss_modular import BYOL_loss
import pandas as pd
import csv
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
import json
from tqdm import tqdm
import numpy as np
from glob import glob
from scipy.io import loadmat
import time
import torch.nn as nn
import numpy
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models import resnet50
import itertools
from itertools import product
from PIL import Image
import random
from PIL import ImageFile

input_path = './dataset_images/'
data_dir = input_path + 'AVA_images/'


# ImageFile.LOAD_TRUNCATED_IMAGES = True
class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.fc1 = nn.Linear(2048, 512,bias=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2048,bias=True)

   

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


save_freq = 1
scores = {}

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--level', type=int,
                        default=0, help='Level')
    parser.add_argument('--rps', type=int,
                        default=256, help='Patch size')                   
    parser.add_argument('--device', type=int,
                        default=0, help='Device (GPU ID)')
    parser.add_argument('--k', type=int,
                        default=16, help='Negative samples')
    parser.add_argument('--bs', type=int,
                        default=1, help='Number of scenes per batch')
    parser.add_argument('--tao', type=float,
                        default=0.1, help='Temperature parameter in the contrastive loss')
    parser.add_argument('--allden', action='store_true',
                        help='Use both denominators together in one denominator')

    parser.add_argument('--all_loss', action='store_true',
                        help='Use all negative combinations')
    
    parser.add_argument('--oneanchorperk', action='store_true',
                        help='Only one anchor per k distortions')
    parser.add_argument('--sup', action='store_true',
                        help='Supervised contrastive loss')
    parser.add_argument('--log', action='store_true',
                        help='Test mode for model')
    parser.add_argument('--head', action='store_true',
                        help='Use all negative combinations')
    
    parser.add_argument('--relu', action='store_true',
                        help='Use relu')
    parser.add_argument('--savefolder', type=str,
                        default='./regularized_check/', help='Folder in which the models are saved')
    parser.add_argument('--pt_path', type=str,
                        default='./pre_trained_models/001_syn_pre_5.pth', help='savefolder')
    
    parser.add_argument('--mi_lambda', type=int,
                        default=1000, help='Scaling term for the MI loss')
    parser.add_argument('--mi_lr', type=float,
                        default=1e-7, help='MI encoder learning rate')
    parser.add_argument('--mi_contrastive', action='store_true', 
                        help='MI Loss - Contrastive term')
    parser.add_argument('--mi_start', type=int,
                        default=1, help='MI term starting epoch')
    
    parser.add_argument('--num_epochs', type=int,
                        default=5, help='Number of epochs')
    
    parser.add_argument('--lowpass', action='store_true',
                        help='use current level lowpass')
    optn = parser.parse_args()
    if optn.lowpass:      
        if optn.head:
            optn.saveloc = optn.savefolder  + 'level_' + str(optn.level) + '_lowpass_withhead'
        else:
            optn.saveloc = optn.savefolder  + 'level_' + str(optn.level) + '_lowpass'
    else:
        optn.saveloc = optn.savefolder  + 'level_' + str(optn.level)
    
    if optn.relu:
        optn.saveloc =  optn.savefolder  + 'level_' + str(optn.level) +'_relu'


    if optn.log : 
        os.makedirs(optn.saveloc, exist_ok=True)
        with open(optn.saveloc + '/config.txt', 'w') as f:
            json.dump(optn.__dict__, f, indent=2)
    

    return optn

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn
import math


args = parse_option()


k = args.k
tao = args.tao
bs = args.bs
rps = args.rps
pt_path = args.pt_path
cuda1 = torch.device('cuda:' + str(args.device))


kernelh = Variable(torch.tensor([[[[0.0625, 0.2500, 0.3750, 0.2500, 0.0625]]]], device=cuda1), requires_grad=False)
kernelv = Variable(torch.tensor([[[[0.0625], [0.2500], [0.3750], [0.2500], [0.0625]]]], device=cuda1), requires_grad=False)

kernel = kernelv*kernelh*4
kernel1 = kernelv*kernelh

ker00 = kernel[:,:,0::2,0::2]
ker01 = kernel[:,:,0::2,1::2]
ker10 = kernel[:,:,1::2,0::2]
ker11 = kernel[:,:,1::2,1::2]

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))

    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def BuildLapPyr(im):
    gpyr2 = pyrReduce(im)
    gpyr3 = pyrReduce(gpyr2)
    gpyr4 = pyrReduce(gpyr3)
    # gpyr5 = pyrReduce(gpyr4)
    
    sub1 = im - pyrExpand(gpyr2)
    sub2 = gpyr2 - pyrExpand(gpyr3)
    sub3 = gpyr3 - pyrExpand(gpyr4)
    
        
    # return sub1, sub2, sub3, sub4 ,gpyr5
    return sub1, sub2, sub3, gpyr4


def pyrReduce(im):
    if im.size(3) % 2 != 0:
        im = torch.cat((im,im[:,:,:,-1:]),dim=-1)
    if im.size(2) % 2 !=0:
        im = torch.cat((im,im[:,:,-1:,:]),dim=-2)              
    
    
    im_out = torch.zeros(im.size(0),3,int(im.size(2)/2),int(im.size(3)/2), device=cuda1)
    
   
    for k in range(3):
        
        temp = im[:,k,:,:].unsqueeze(dim=1)
        
        im_cp = torch.cat((temp[:,:,:,0].unsqueeze(dim=3), temp[:,:,:,0].unsqueeze(dim=3), temp), dim=3) # padding columns
        im_cp = torch.cat((im_cp, im_cp[:,:,:,-1].unsqueeze(dim=3), im_cp[:,:,:,-1].unsqueeze(dim=3)), dim=3) # padding columns
        
        im_bp = torch.cat((im_cp[:,:,0,:].unsqueeze(dim=2), im_cp[:,:,0,:].unsqueeze(dim=2), im_cp), dim=2) # padding columns
        im_bp = torch.cat((im_bp, im_bp[:,:,-1,:].unsqueeze(dim=2), im_bp[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
        
        im1 = F.conv2d(im_bp, kernel1, padding = [0,0], groups=1)
        im_out[:,k,:,:] = im1[:,:,0::2,0::2]
    

         
    
    
    return im_out                 


def pyrExpand(im):
    


    
    out = torch.zeros(im.size(0),im.size(1),im.size(2)*2,im.size(3)*2, device=cuda1, dtype=torch.float32)
    
    for k in range(3):
        
        temp = im[:,k,:,:]
        temp = temp.unsqueeze(dim=1)
                       
        im_c1 = torch.cat((temp, temp[:,:,:,-1].unsqueeze(dim=3)), dim=3) 
        im_c1r1 = torch.cat((im_c1, im_c1[:,:,-1,:].unsqueeze(dim=2)), dim=2) 
                
        im_r2 = torch.cat((temp[:,:,0,:].unsqueeze(dim=2), temp), dim=2) # padding columns
        im_r2 = torch.cat((im_r2, im_r2[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
        
        im_r2c1 = torch.cat((im_r2, im_r2[:,:,:,-1].unsqueeze(dim=3)), dim=3) 
                
        im_c2 = torch.cat((temp[:,:,:,0].unsqueeze(dim=3), temp), dim=3) # padding columns
        im_c2 = torch.cat((im_c2, im_c2[:,:,:,-1].unsqueeze(dim=3)), dim=3) # padding columns
        
        im_c2r1 = torch.cat((im_c2, im_c2[:,:,-1,:].unsqueeze(dim=2)), dim=2) 
                
        im_r2c2 = torch.cat((im_c2[:,:,0,:].unsqueeze(dim=2), im_c2), dim=2) # padding columns
        im_r2c2 = torch.cat((im_r2c2, im_r2c2[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
                
        im_00 = F.conv2d(im_r2c2, ker00, padding = [0,0], groups=1)
        im_01 = F.conv2d(im_r2c1, ker01, padding = [0,0], groups=1)
        im_10 = F.conv2d(im_c2r1, ker10, padding = [0,0], groups=1)
        im_11 = F.conv2d(im_c1r1, ker11, padding = [0,0], groups=1)
        
        out[:,k,0::2,0::2] = im_00
        out[:,k,1::2,0::2] = im_10
        out[:,k,0::2,1::2] = im_01
        out[:,k,1::2,1::2] = im_11
                 
    return out

def pad_diff(term1,term2):
    max_h = max([term1.size(-2),term2.size(-2)])
    max_w = max([term1.size(-1),term2.size(-1)])
    w_diff1 = max_w - term1.size(-1)
    h_diff1 = max_h - term1.size(-2)
    if w_diff1 != 0 :
        for x in range(w_diff1):
            term1 = torch.cat((term1,term1[:,:,:,-1:]),dim=-1)
    if h_diff1 != 0 :
        for x in range(h_diff1):
            term1 = torch.cat((term1,term1[:,:,-1:,:]),dim=-2)

    w_diff2 = max_w - term2.size(-1)
    h_diff2 = max_h - term2.size(-2)
    if w_diff2 != 0 :
        for x in range(w_diff2):
            term2 = torch.cat((term2,term2[:,:,:,-1:]),dim=-1)
    if h_diff2 != 0 :
        for x in range(h_diff2):
            term2 = torch.cat((term2,term2[:,:,-1:,:]),dim=-2)

    diff = term1 - term2
    return diff

class US_Data(Dataset):
    def __init__(self, data_dir, k, transform=None):
        filenames = os.listdir(data_dir)
        # filenames = [i for i in filenames if '.jpg' in i]
        # filenames = pd.read_csv('./sampled_AVA.csv',names=['tp','image_name'])
        # filenames = list(filenames['image_name'])
        # filenames = filenames[1:]
        
        # filenames = random.sample(filenames,10000)
        # filenamesdf = pd.DataFrame(filenames)
        # filenamesdf.to_csv('./sampled_AVA.csv')
        self.pathnames = [os.path.join(data_dir, f) for f in filenames]
        
        # create empty list to store the combinations
        # self.pathnames = list(itertools.combinations(pathnames, 2))

        self.transform = transform
        self.k = k

    def __len__(self):
        return len(self.pathnames)

    def __getitem__(self, idx):

        img = self.pathnames[idx]
        
        curr_img = Image.open(img)
        curr_img = self.transform(curr_img)
        if curr_img.size(0)==1:
           curr_img = curr_img.repeat(3, 1, 1)


        first = curr_img.unsqueeze(dim=0)
        first = first.cuda(cuda1)
        if args.level == 0:
            first_pass = first
        
        elif args.level == 1:
            first_temp = pyrReduce(first)  

            
            if args.lowpass:
                first_pass = first_temp
                # first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)                
            else:           
                # print(first.size(),pyrExpand(first_temp).size())    
                # exit(0)
                        
                first_pass = pad_diff(first,pyrExpand(first_temp))

        elif args.level == 2:
            gpyr2 = pyrReduce(first)
            first_temp = pyrReduce(gpyr2)            
            if args.lowpass:
                first_pass = first_temp
                # first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)                
            else:                       
                first_pass = pad_diff(gpyr2 , pyrExpand(first_temp))
    
        elif args.level == 3:
            gpyr2 = pyrReduce(first)
            gpyr3 = pyrReduce(gpyr2)
            first_temp = pyrReduce(gpyr3)
            if args.lowpass:
                first_pass = first_temp
                first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)                
            else:                       
                first_pass = pad_diff(gpyr3 ,pyrExpand(first_temp)  )
        
             
        if first_pass.size(2)%2:
            first_pass = torch.cat((first_pass,first_pass[:,:,-1:,:]),dim=-2)
        if first_pass.size(3)%2:
            first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)

        batch = first_pass
        

        global ps 
        ps = min(int(batch.size(2)/2),int(batch.size(3)/2))
       
        ###twopatchsampler############
        f = randint(1, 3)
        if f == 1:
            half1, half2 = torch.split(
                batch, int(batch.size(2)/2), dim=2)            

        else:
            half1, half2 = torch.split(
                batch, int(batch.size(3)/2), dim=3)

        crop = transforms.RandomCrop(ps)
        p1_batch = crop(half1)
        p2_batch = crop(half2)
        p1_batch = transforms.Resize(rps)(p1_batch)
        p2_batch = transforms.Resize(rps)(p2_batch)
        ps = rps
        return p1_batch, p2_batch

class ContentNetwork(nn.Module):
    def __init__(self):
        super(ContentNetwork, self).__init__()
        self.net = resnet50(pretrained=True)

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class MIModel(nn.Module):
    def __init__(self, z_len=2048, xc_len=1000):
        super(MIModel, self).__init__()
        self.dense1 = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_features=z_len, out_features=xc_len, bias=True),
            nn.Linear(in_features=xc_len, out_features=xc_len, bias=True),
        )
        self.dense2 = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_features=z_len, out_features=xc_len, bias=True),
            nn.Linear(in_features=xc_len, out_features=xc_len, bias=True),
            nn.Tanh(),
        )

    def forward(self, feat):
        return self.dense1(feat), self.dense2(feat)


class MILoss(nn.Module):

    def __init__(self, model, mi_params):
        super(MILoss, self).__init__()
        self.content_net = ContentNetwork()
        self.model = model
        for param in self.content_net.parameters():
            param.requires_grad = False
        self.content_net.eval()
        self.mi_est_model = MIModel(z_len=mi_params['z_len'], xc_len=1000)
        self.mi_opt = torch.optim.Adam(self.mi_est_model.parameters(), lr=mi_params['q_opt_lr'], weight_decay=0.1)
        self.q_contrastive_loss = mi_params['q_contrastive_loss']

    def forward(self, X, z_cat):
        self.mi_est_model.train()

        with torch.no_grad():
            z = self.model(X).squeeze()
        
        mu, logvar = self.mi_est_model(z)
        var = torch.exp(logvar)
        xc = self.content_net(X)

        if self.q_contrastive_loss:
            mu_1 = torch.unsqueeze(mu, dim=0)
            var_1 = torch.unsqueeze(var, dim=0)
            xc_1 = torch.unsqueeze(xc, dim=1)
            mi_est_loss = torch.mean(torch.sum(((xc - mu) ** 2) / var, dim=1)) - torch.mean(
                torch.sum(((xc_1 - mu_1) ** 2) / var_1, dim=2))
        else:
            mi_est_loss = torch.mean(torch.sum(((xc - mu) ** 2) / var, dim=1)) + torch.mean(
                torch.sum(logvar, dim=1))

        self.mi_opt.zero_grad()
        mi_est_loss.backward()
        self.mi_opt.step()

        self.mi_est_model.eval()

        mu, logvar = self.mi_est_model(z_cat)
        var = torch.exp(logvar)
        mu_1 = torch.unsqueeze(mu, dim=0)
        var_1 = torch.unsqueeze(var, dim=0)
        xc_1 = torch.unsqueeze(xc, dim=1)
        mi_2lb_l = -torch.mean(torch.sum(((xc - mu) ** 2) / var, dim=1)) + torch.mean(
            torch.sum(((xc_1 - mu_1) ** 2) / var_1, dim=2))

        return mi_2lb_l

    def ramp(self, iteration_num):
        return min(1, max(0, (iteration_num - self.ramp_range[0]) / (self.ramp_range[1] - self.ramp_range[0])))

transform_train = transforms.Compose([transforms.ToTensor()])
train_dataloader = DataLoader(US_Data(data_dir, k, transform_train),
                              batch_size=k, shuffle=True,drop_last=True)

model2 = Predictor()

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

res = resnet50()
modules = list(res.children())[:-1]
model = nn.Sequential(*modules)
model.load_state_dict(torch.load(pt_path, map_location='cpu'))

count = 0
for param in model.children():
        count +=1
        if count < 7: 
            param.requires_grad = False
count = 0


for param in model.children():
        count +=1
        if count < 3: 
            param.requires_grad = True
            

indmat = createindmat(k,bs)
model.train()
model = model.cuda(cuda1)
model2.train()
model2 = model2.cuda(cuda1)
opt1 = optim.Adam(model.parameters(), lr=1e-3)
opt2 = optim.Adam(model2.parameters(), lr=1e-4)


criterion = BYOL_loss()
eps =1e-8
for epoch in range(args.num_epochs):
    if epoch == args.mi_start:
        mi_params = {'q_opt_lr': args.mi_lr,
                    'q_contrastive_loss': args.mi_contrastive,
                    'z_len': 2048}     # Size of the feature vector
        mi_loss_fn = MILoss(model, mi_params).cuda(cuda1)
    if epoch >= args.mi_start:
        lamb = args.mi_lambda
    else:
        lamb = 0
    epoch_loss = 0
    start_time = time.time()
    for n_count, batch in enumerate(tqdm(train_dataloader)):
        batchtime = time.time()
        opt1.zero_grad()
        opt2.zero_grad()
        p1_batch, p2_batch = Variable(batch[0]), Variable(
            batch[1])
        stacked = torch.stack((p1_batch, p2_batch), dim=2)  
    
        stacked = stacked.view(bs*2*k,3,ps,ps)

        passsamples = stacked.cuda(cuda1)

        feats = model(passsamples)
        preds = model2(feats.squeeze())
        feats = feats.view(bs,k,2,-1).squeeze()
        preds = preds.view(bs,k,2,-1).squeeze()
        z1 = feats[:,0,:]
        z2 = feats[:,1,:]
        p1 = preds[:,0,:]
        p2 = preds[:,1,:]

        
        if lamb !=0:
            z_cat = torch.cat((z1, z2), dim=0)            
            miloss = lamb*mi_loss_fn(passsamples, z_cat)
            closs = criterion(z1,z2,p1,p2)    
            loss = closs + miloss
        else:
            loss = criterion(z1,z2,p1,p2)
        
        epoch_loss += loss.item()
        loss.backward()        
        opt1.step()
        opt2.step()
        batchtimed = time.time() - batchtime
        
    elapsed_time = time.time() - start_time
    print('epoch = %4d , loss = %4.4f , time = %4.2f s' %
        (epoch + 1, epoch_loss / n_count, elapsed_time))
    if (epoch + 1) % save_freq == 0:
        torch.save(model.state_dict(), os.path.join(
            args.saveloc, 'model_%03d.pth' % (epoch+1)))
        torch.save(model2.state_dict(), os.path.join(
            args.saveloc, 'model2_%03d.pth' % (epoch+1)))
    
