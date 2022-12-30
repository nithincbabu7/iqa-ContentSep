from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from os.path import join
from tqdm import tqdm


class LIVE_Challenge(Dataset):
    def __init__(self, img_dir, data_loc, 
                 training_images=False, 
                 resize_to_500=True,
                 transform=transforms.ToTensor(), 
                 args=None):
        self.img_dir = img_dir
        self.transform = transform
        self.args = args
        if resize_to_500:
            self.r500 = transforms.Resize((500, 500))
        else:
            self.r500 = None
        
        self.data = pd.read_csv(data_loc, index_col=[0])
        self.data = self.data.astype({'im_loc': str, 'mos': np.float32})
        if not training_images:
            self.data = self.data[['trainingImages' not in self.data['im_loc'].iloc[i] for i in range(len(self.data))]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_loc = self.data.iloc[idx]['im_loc']
        x = Image.open(join(self.img_dir, im_loc))
        x = self.transform(x)
        if self.r500:
            if im_loc == '1024.JPG' or im_loc == '1113.JPG':
                x = self.r500(x)
            
        return x, self.data.iloc[idx]['mos'], im_loc


class KONIQ_10k(Dataset):
    def __init__(self, img_dir, data_loc,
                 transform=transforms.ToTensor(), 
                 args=None):
        self.img_dir = img_dir
        self.transform = transform
        self.args = args

        self.data = pd.read_csv(data_loc, index_col=['index'])
        self.data = self.data.astype({'im_loc': str, 'mos': np.float32})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_loc = self.data.iloc[idx]['im_loc']
        x = Image.open(join(self.img_dir, im_loc))
        x = self.transform(x)
            
        return x, self.data.iloc[idx]['mos'], im_loc


class LIVE_FB(Dataset):
    def __init__(self, img_dir, data_loc,
                 transform=transforms.ToTensor(), 
                 resize=512,
                 args=None):
        self.img_dir = img_dir
        self.args = args
        self.resize = resize
        if self.resize:
            self.transform = transforms.Compose([transform, transforms.Resize((resize, resize))]) 

        self.data = pd.read_csv(data_loc, index_col=['index'])
        self.data = self.data.astype({'im_loc': str, 'mos': np.float32})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_loc = self.data.iloc[idx]['im_loc']
        x = Image.open(join(self.img_dir, im_loc))
        if x.mode != 'RGB':
            x = x.convert('RGB')
        x = self.transform(x)
            
        return x, self.data.iloc[idx]['mos'], im_loc


class CID2013(Dataset):
    def __init__(self, img_dir, data_loc,
                 transform=transforms.ToTensor(),
                 args=None):
        self.img_dir = img_dir
        self.args = args
        self.transform = transform

        self.data = pd.read_csv(data_loc, index_col=[0])
        self.data = self.data.astype({'im_loc': str, 'mos': np.float32})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_loc = self.data.iloc[idx]['im_loc']
        x = Image.open(join(self.img_dir, im_loc))
        x = self.transform(x)
            
        return x, self.data.iloc[idx]['mos'], im_loc


if __name__ == '__main__':
    # img_dir = './dataset_images/LIVEC/Images/'
    # data_loc = './datasets/LIVEC.csv'
    # dataset = LIVE_Challenge(img_dir, data_loc)

    # img_dir = './dataset_images/KONIQ/1024x768/'
    # data_loc = './datasets/KONIQ.csv'
    # dataset = KONIQ_10k(img_dir, data_loc)

    # img_dir = './dataset_images/LIVEFB/'
    # data_loc = './datasets/LIVEFB.csv'
    # dataset = LIVE_FB(img_dir, data_loc)

    img_dir = './dataset_images/CID2013/'
    data_loc = './datasets/CID2013.csv'
    dataset = CID2013(img_dir, data_loc)


    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    print(len(dataset), len(loader))
    # x, y, name = next(iter(loader))

    for (batch, (x,y,name)) in enumerate(tqdm(loader)):
        pass
