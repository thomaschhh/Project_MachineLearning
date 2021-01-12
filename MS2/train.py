#!/usr/bin/env python3
#$ -l cuda=1 # remove this line when no GPU is needed!
#$ -q all.q # do not fill the qlogin queue
#$ -cwd # start processes in current working directory
#$ -V # provide environment variables to processes
#$ -o output.txt
#$ -e error.txt

import sys
sys.path.append('/home/pml_16/MS2')
import numpy as np
from data_loader import load_data
import Models
import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
from preprocessing import compute_features, preprocessing, clustering, cluster_assign
from utils import UnifLabelSampler, AverageMeter
from tqdm import tqdm
from visualization import plot_loss_acc, show_img
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--path',  help='path to dataset',type = str, default = '/home/space/datasets/imagenet/2012/train_set_small')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--k', '--nmb_cluster', type=int, default=50,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--ep', type=int, default=100,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--bs', default=50, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()

def train(dataLoader, model, crit, optimizer, epoch, lr, wd):
    for i, (input_tensor, target) in enumerate(dataLoader):
        print(f'trained {i}-th feature')
        losses = AverageMeter()
        accuracies = AverageMeter()
        # switch to train mode
        model.train()
        # create an optimizer for the last fc layer
        optimizer_tl = torch.optim.SGD(
            model.top_layer.parameters(),
            lr=lr,
            weight_decay=10**wd,
        )
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())
        #input_var = torch.autograd.Variable(input_tensor)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
      
        loss = crit(output, target_var)
        acc = torch.sum(output == target_var) 
       
        losses.update(loss.data, input_tensor.size(0))
        accuracies.update(acc.data,input_tensor.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_tl.step()
        
    return losses.avg, accuracies.avg

def validate(dataloader, model, crit):
    #test
    model.eval()
   
    for i, (input_tensor, target) in enumerate(dataloader):
        losses = AverageMeter()
        accuracies = AverageMeter()
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())
       
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        print(output)
        print(target_var)
        loss = crit(output, target_var) 
        acc = torch.sum(output == target_var)
        losses.update(loss.data, input_tensor.size(0))
        accuracies.update(acc.data,input_tensor.size(0))
    return losses.avg, accuracies.avg

def main(args):
    # fix random seeds
    print('start training')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    now = datetime.now()
    # load the data
    dataloader, dataset_train, dataloader_val, dataset_val = load_data(args.path, args.bs, train_ratio = 0.9, test_ratio = 0.1)
    #load vgg
    model = Models.__dict__["vgg16"](args.sobel) # pretrained weights?
    fd = int(model.top_layer.weight.size()[1]) 
    model.top_layer = None # why? do we need it here?
    model.features = torch.nn.DataParallel(model.features)    
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=10**args.wd,
       )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()
    losses = np.zeros(args.ep) # loss per epoch, array of size ep x 1
    accuracies = np.zeros(args.ep)
    losses_val = np.zeros(args.ep)
    accuracies_val = np.zeros(args.ep)
    labels = [573, 671] # move to another location, maybe outside for-loop, outside training method
    
    # for all epochs
    for epoch in range(args.ep):
        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) # The actual classifier seems missing here, why are just the children added to a list?
        # get the features for the whole dataset
        
        features = compute_features(dataloader, model, len(dataset_train), args.bs, labels)
        features_val = compute_features(dataloader_val, model, len(dataset_val), args.bs, labels)
        print('PCA')
        pre_data = preprocessing(model, features)
        pre_data_val = preprocessing(model, features_val)
        print('clustering')
        clus_data, images_list = clustering(pre_data, args.k)
        clus_data_val, images_list_val = clustering(pre_data_val, args.k)
        # pseudo labels
        print('train pseudolabels')
        train_dataset = cluster_assign(images_list, dataset_train)
        val_dataset = cluster_assign(images_list_val, dataset_val)
        len_d = len(train_dataset)
        len_val = len(val_dataset)
        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign * len_d),images_list)
        sampler2 = UnifLabelSampler(int(args.reassign * len_val),images_list_val)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.bs,
            sampler=sampler,
            pin_memory=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.bs,
            sampler=sampler2,
            pin_memory=True,
        )
        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(images_list))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()
        # train network with clusters as pseudo-labels
        
        losses[epoch], accuracies[epoch] = train(train_dataloader, model, criterion, optimizer, epoch, args.lr, args.wd)
        print(f'epoch {epoch} ended with loss {losses[epoch]}')
        #losses_val[epoch], accuracies_val[epoch] = validate(val_dataloader, model, criterion)
        plot_loss_acc(losses[0:epoch],losses[0:epoch], accuracies[0:epoch], accuracies[0:epoch], epoch, now)
    
    
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
