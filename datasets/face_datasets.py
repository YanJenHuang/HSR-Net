import os
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def load_dataset(root='datasets', db_name='morph2', transform=[None, None]):

    if db_name not in ['imdb', 'wiki', 'morph2', 'megaage_asian', 'morph2_context']:
        raise ValueError('"db_name" error, "imdb", "wiki", "morph2", "megaage_asian".')
    
    if transform[0] == None:
        transform_train = transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(20)], 0.25),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=None, scale=None, shear=0.2, resample=False, fillcolor=0)], 0.25),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=[0.2,0.2], scale=None, shear=None, resample=False, fillcolor=0)], 0.25),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=None, scale=[0.8,1.2], shear=None, resample=False, fillcolor=0)], 0.25),
            transforms.ToTensor()
        ])
    else:
        transform_train = transform[0]
    
    if transform[1] == None:
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transform_test = transform[1]
    
    if db_name in ['imdb', 'wiki', 'morph2', 'megaage_asian']:
        trainset = FaceDataset(root=root,db_name=db_name,train=True,transform=transform_train)
        validset = FaceDataset(root=root,db_name=db_name,train=False,transform=transform_test)
    elif db_name in ['morph2_context']:
        trainset = FaceDatasetContext(root=root,db_name=db_name,train=True,transform=transform_train)
        validset = FaceDatasetContext(root=root,db_name=db_name,train=False,transform=transform_test)
    return trainset, validset

''' Examples
transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomRotation(20)], 0.25),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=None, scale=None, shear=0.2, resample=False, fillcolor=0)], 0.25),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=[0.2,0.2], scale=None, shear=None, resample=False, fillcolor=0)], 0.25),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=None, scale=[0.8,1.2], shear=None, resample=False, fillcolor=0)], 0.25),
    transforms.ToTensor()
])
transform_tensor = transforms.Compose([
    transforms.ToTensor()
])

imdb_trainset = FaceDataset(root='datasets', db_name='imdb', train=True, transform=transform, seed=1)
imdb_validset = FaceDataset(root='datasets', db_name='imdb', train=False, transform=transform, seed=1)
wiki_trainset = FaceDataset(root='datasets', db_name='wiki', train=True, transform=transform, seed=1)
wiki_validset = FaceDataset(root='datasets', db_name='wiki', train=False, transform=transform, seed=1)
morph2_trainset = FaceDataset(root='datasets', db_name='morph2', train=True, transform=transform, seed=1)
morph2_validset = FaceDataset(root='datasets', db_name='morph2', train=False, transform=transform, seed=1)
'''
class FaceDataset(Dataset):
    """ Face dataset
    IMDB
    WIKI
    Morph2
    MegaAgeAsian
    """
    def __init__(self, root, db_name, train=True, transform=None, split_ratio=0.8, seed=1):
        self.root = root
        self.db_name = db_name
        self.train = train
        self.transform = transform
        self.split_ratio = split_ratio
        self.seed = seed
        self.npz_file_path = None
        
        self.id = None
        self.image = None
        self.age = None
        self.image_path = None
        self.image_size = None
        
        db_list = ['imdb', 'wiki', 'morph2', 'megaage_asian']
        if self.db_name not in db_list:
            raise ValueError('db_name should be {}, instead {}'.format(db_list, self.db_name))
        else:
            if self.db_name in ['imdb', 'wiki', 'morph2']:
                self.npz_file_path = os.path.join(root, '{}.npz'.format(self.db_name))
            elif self.db_name in ['megaage_asian']:
                if self.train == True:
                    self.npz_file_path = os.path.join(root, 'megaage_asian/megaage_train.npz')
                else:
                    self.npz_file_path = os.path.join(root, 'megaage_asian/megaage_test.npz')
                
            if os.path.exists(self.npz_file_path) == True:
                print('@@@ Start loading dataset: "{}"...'.format(self.npz_file_path))
            else:
                raise ValueError('npz_file "{}" not exists.'.format(self.npz_file_path))

        # Start loading dataset
        self.id, self.image, self.age, self.image_path, self.image_size = self.load_npz_file()
        
        if self.db_name in ['imdb', 'wiki', 'morph2']:
            print('@@@ Using seed={} to shuffle dataset...'.format(self.seed))
            print('>>> Dataset will be splitted into trainset({:.2f}) and validset({:.2f})...'.format(self.split_ratio, 1-self.split_ratio))
        
            np.random.seed(self.seed)
            seed_idx = np.random.permutation(len(self))
            num_train_example = int(len(self)*self.split_ratio)
            if self.train == True:
                print('>>> Return Trainset({:.2f})'.format(self.split_ratio))
                self.id = self.id[seed_idx][:num_train_example]
                self.image = self.image[seed_idx][:num_train_example]
                self.age = self.age[seed_idx][:num_train_example]
                self.image_path = self.image_path[seed_idx][:num_train_example]
            else:
                print('>>> Return Validset({:.2f})'.format(1-self.split_ratio))
                self.id = self.id[seed_idx][num_train_example:]
                self.image = self.image[seed_idx][num_train_example:]
                self.age = self.age[seed_idx][num_train_example:]
                self.image_path = self.image_path[seed_idx][num_train_example:]
        
        elif self.db_name in ['megaage_asian']:
            print('@@@ Seed={} is not used for shuffling dataset...'.format(self.seed))
            print('>>> Dataset already been splitted into trainset({}) and validset({}) by two npz files...'.format(40000, 3945))

            np.random.seed(self.seed)
            seed_idx = np.random.permutation(len(self))
            if self.train == True:
                print('>>> Return Trainset({})'.format(len(self)))
                self.id = self.id[seed_idx]
                self.image = self.image[seed_idx]
                self.age = self.age[seed_idx]
                self.image_path = self.image_path[seed_idx]
            else:
                print('>>> Return Validset({:.2f})'.format(len(self)))
                # None

    def __getitem__(self, idx):
        img = self.image[idx]
        age = self.age[idx]
                
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        
        return img, age
    
    def show_sample(self, img_type='transform'):
        if img_type not in ['transform', 'original']:
            print('type should be "transform" or "original", instead get {}'.format(img_type))
        
        idx = np.random.randint(low=0, high=len(self))
        if img_type == 'transform':
            transform_toPILImage = transforms.Compose([
                transforms.ToPILImage()
            ])
            img, age = self.__getitem__(idx)
            img = transform_toPILImage(img)
        else:
            img = self.image[idx]
            age = self.age[idx]

        plt.imshow(img)
        plt.title(label=str(age))
        plt.show()
        
    def __len__(self):
        return len(self.image_path)
    
    def load_npz_file(self):
        npz_file = np.load(self.npz_file_path)
        print('npz_file keys: {}'.format(npz_file.files))
        return npz_file['id'], npz_file['image'], npz_file['age'], npz_file['img_path'], npz_file['img_size']

    def show_dataset_distribution(self):
        if self.train == True:
            print('{} trainset has {} examples'.format(self.db_name, len(self)))
        else:
            print('{} validset has {} examples'.format(self.db_name, len(self)))
        
        plt.hist(self.age, bins=100, range=(0,100), edgecolor='black')
        plt.show()

        
# Context FaceDataset        
class FaceDatasetContext(Dataset):
    """ Face dataset Context
    Only support Morph2 dataset
    """
    def __init__(self, root, db_name, train=True, transform=None, split_ratio=0.8, seed=1):
        self.root = root
        self.db_name = db_name
        self.train = train
        self.transform = transform
        self.split_ratio = split_ratio
        self.seed = seed
        self.npz_file_path = None
        
        self.id = None
        self.image = None
        self.age = None
        self.image_path = None
        self.image_size = None
        
        db_list = ['morph2_context']
        if self.db_name not in db_list:
            raise ValueError('db_name should be {}, instead {}'.format(db_list, self.db_name))
        else:
            self.npz_file_path = os.path.join(root, '{}.npz'.format(self.db_name))                
            if os.path.exists(self.npz_file_path) == True:
                print('@@@ Start loading dataset: "{}"...'.format(self.npz_file_path))
            else:
                raise ValueError('npz_file "{}" not exists.'.format(self.npz_file_path))

        # Start loading dataset
        self.id, self.image, self.age, self.image_path, self.image_size = self.load_npz_file()
        
        print('@@@ Using seed={} to shuffle dataset...'.format(self.seed))
        print('>>> Dataset will be splitted into trainset({:.2f}) and validset({:.2f})...'.format(self.split_ratio, 1-self.split_ratio))
        
        np.random.seed(self.seed)
        seed_idx = np.random.permutation(len(self))
        num_train_example = int(len(self)*self.split_ratio)
        if self.train == True:
            print('>>> Return Trainset({:.2f})'.format(self.split_ratio))
            self.id = self.id[seed_idx][:num_train_example]
            self.image = self.image[seed_idx][:num_train_example]
            self.age = self.age[seed_idx][:num_train_example]
            self.image_path = self.image_path[seed_idx][:num_train_example]
        else:
            print('>>> Return Validset({:.2f})'.format(1-self.split_ratio))
            self.id = self.id[seed_idx][num_train_example:]
            self.image = self.image[seed_idx][num_train_example:]
            self.age = self.age[seed_idx][num_train_example:]
            self.image_path = self.image_path[seed_idx][num_train_example:]
            
    def __getitem__(self, idx):
        def _init_seed_(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        img = self.image[idx]
        age = self.age[idx]
        
        high_img = img[0]
        medium_img = img[1]
        low_img = img[2]
        if self.transform is not None:
            seed = np.random.randint(low=0,high=2**32-1)
            
            high_img = Image.fromarray(high_img)
            medium_img = Image.fromarray(medium_img)
            low_img = Image.fromarray(low_img)
            
            _init_seed_(seed)
            high_img = self.transform(high_img)
            _init_seed_(seed)
            medium_img = self.transform(medium_img)
            _init_seed_(seed)
            low_img = self.transform(low_img)
            
            img = (high_img, medium_img, low_img)
        return img, age
        
    def __len__(self):
        return len(self.image_path)
    
    def load_npz_file(self):
        npz_file = np.load(self.npz_file_path)
        print('npz_file keys: {}'.format(npz_file.files))
        return npz_file['id'], npz_file['image'], npz_file['age'], npz_file['img_path'], npz_file['img_size']

    def show_dataset_distribution(self):
        if self.train == True:
            print('{} trainset has {} examples'.format(self.db_name, len(self)))
        else:
            print('{} validset has {} examples'.format(self.db_name, len(self)))
        
        plt.hist(self.age, bins=100, range=(0,100), edgecolor='black')
        plt.show()
