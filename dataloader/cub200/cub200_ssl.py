import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

identity = lambda x: x

class CUB200(Dataset):

    def __init__(self, root='./filelists', train=True,
                 index_path=None, index=None, noise_rate=0., n_shot=5, n_query=0, incremental=False):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self._pre_operate(self.root)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            sub_data_loader_params = dict(batch_size=n_shot + n_query,
                                          shuffle=True,
                                          num_workers=0,  # use main thread only or may receive multiple batches
                                          pin_memory=False)

            # if incremental:
            #     self.transform = transforms.Compose([
            #         transforms.Resize((224, 224)),
            #         # transforms.CenterCrop(224),
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #     ])
            # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            # if base_sess:
            #     self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            # else:
            if index is not None:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            sub_data_loader_params = dict(batch_size=n_shot + n_query,
                                          shuffle=False,
                                          num_workers=0,  # use main thread only or may receive multiple batches
                                          pin_memory=False)
            if index is not None:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)

        self.cl_list = np.unique(self.targets).tolist()
        self.sub_meta = {}
        self.sub_dataloader = []

        for cl in self.cl_list:
            self.sub_meta[cl] = []
        for x, y in zip(self.data, self.targets):
            self.sub_meta[y].append(x)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, self.transform, identity, noise_rate)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))


    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.cl_list)

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))


class SubDataset:
    def __init__(self, sub_meta, cl, transform, identity, noise_rate):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = identity
        self.noise_rate = noise_rate

    def __getitem__(self, i):
        # print( '%d -%d' %(self.cl,i))
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        if self.noise_rate > 0.:
            if np.random.random() > (1 - self.noise_rate):
                # img = np.array(img)
                # img = (img + np.random.randint(0, 255, size=img.shape)) // 2
                img = np.array(img)
                img = np.random.randint(0, 255, size=img.shape)
                img = Image.fromarray(img.astype('uint8'))
        img = self.transform(img)
        target = self.target_transform(self.cl)
        # print("path:{}, label:{}".format(image_path, self.cl))
        return img, target

    def __len__(self):
        return len(self.sub_meta)
