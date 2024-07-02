import numpy as np
import torch
from dataloader.sampler import CategoriesSampler, EpisodicBatchSampler

def set_up_datasets(args):

    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset

        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset

        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'PlantVillage':
        import dataloader.plantvillage.plantvillage as Dataset

        args.base_class = 20
        args.num_classes=38
        args.way = 3
        args.shot = 5
        args.sessions = 7

    args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    class_index = np.arange(args.base_class)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'PlantVillage':
        trainset = args.Dataset.PlantVillage(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.PlantVillage(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader



def get_base_dataloader_meta(args):
    txt_path = args.dataroot+"/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)
    if args.dataset == 'PlantVillage':
        trainset = args.Dataset.PlantVillage(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.PlantVillage(root=args.dataroot, train=False,
                                            index=class_index)

    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    txt_path = args.dataroot+"/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'PlantVillage':
        trainset = args.Dataset.PlantVillage(root=args.dataroot, train=True,
                                       index_path=txt_path)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'PlantVillage':
        testset = args.Dataset.PlantVillage(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_ssl_dataloader(args):
    dataroot = args.dataroot
    train_path = dataroot +'/index_list/' +args.dataset + '/session_' + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)

    if args.dataset == 'cub200':
        import dataloader.cub200.cub200_ssl as SSLDataset
        trainset = SSLDataset.CUB200(root=dataroot, train=True, index_path=train_path, n_shot=args.SSL_shot, n_query=args.SSL_query)
        testset = SSLDataset.CUB200(root=dataroot, train=False, index=class_index, n_shot=args.SSL_shot, n_query=args.SSL_query)

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet_ssl as SSLDataset

        trainset = SSLDataset.MiniImageNet(root=dataroot, train=True, index_path=train_path, n_shot=args.SSL_shot, n_query=args.SSL_query)
        testset = SSLDataset.MiniImageNet(root=dataroot, train=False, index=class_index, n_shot=args.SSL_shot, n_query=args.SSL_query)

    if args.dataset == 'PlantVillage':
        import dataloader.plantvillage.plantvillage_ssl as SSLDataset
        trainset = SSLDataset.PlantVillage(root=dataroot, train=True, index_path=train_path, n_shot=args.SSL_shot, n_query=args.SSL_query)
        testset = SSLDataset.PlantVillage(root=dataroot, train=False, index=class_index, n_shot=args.SSL_shot, n_query=args.SSL_query)

    train_sampler = EpisodicBatchSampler(len(trainset), n_way=args.SSL_way, n_episodes=100)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8,
                                              pin_memory=True)

    test_sampler = EpisodicBatchSampler(len(testset), n_way=args.SSL_way, n_episodes=500)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=8,
                                              pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list