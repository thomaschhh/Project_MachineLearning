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

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--path',  help='path to dataset',type = str, default = '/home/space/datasets/imagenet/2012/train_set_small')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--k', '--nmb_cluster', type=int, default=5,
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
    parser.add_argument('--ep', type=int, default=30,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--bs', default=9, type=int,
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
        if i ==0:
            show_img(input_tensor[0:9], label = target[0:9])
        
        losses = AverageMeter()
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
      #  print(target.clone().detach().cpu().numpy())
        #print(output.clone().detach().cpu().numpy().shape)
        loss = crit(output, target_var)
        # record loss
        #print(loss)
       
        losses.update(loss.data, input_tensor.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_tl.step()
        
    return losses.avg

def test(dataloader, model, crit):
    #test
    model.eval()
    test_loss = 0
    for i, (input_tensor, target) in enumerate(dataloader):
        output = model(input_tensor)
        test_loss = crit(output, target) 
    return test_loss

#main method
def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # load the data
    dataloader, dataset_train, testL, dataset_test = load_data(args.path, args.bs, train_ratio = 0.1, test_ratio = 0.9)
    #load vgg
    model = Models.__dict__["vgg16"](args.sobel) 
    fd = int(model.top_layer.weight.size()[1]) 
    model.top_layer = None 
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
    losses = np.zeros(args.ep)
    # for all epochs
    for epoch in range(args.ep):
        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        # get the features for the whole dataset
        labels = [573, 671]
        features = compute_features(dataloader, model, len(dataset_train), args.bs, labels)
        pre_data = preprocessing(model, features)
        
        clus_data, images_list= clustering(pre_data,args.k)
        
        # pseudo labels
        train_dataset = cluster_assign(images_list, dataset_train)
        len_d = len(train_dataset)
        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign * len_d),images_list)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.bs,
            sampler=sampler,
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
        
        losses[epoch] = train(train_dataloader, model, criterion, optimizer, epoch, args.lr, args.wd)
        print(f'epoch {epoch} ended with loss {losses[epoch]}')
    
    loss_test = test(testL, model, criterion)
    plot_loss_acc(losses,losses, args.ep)
    print(loss_test)
    
    
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)