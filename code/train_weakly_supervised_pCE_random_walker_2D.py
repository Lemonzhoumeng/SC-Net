import argparse
import logging
import os
import random
import shutil
import sys
import time
from medpy import metric
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset_super import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/pCE_Random_Walker', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='random_walker', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()
def one_hot_encoder(input_tensor,n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


def dice_coefficient(label_true, label_pred, num_classes):
    smooth = 1e-5 
    batch_size = label_true.size(0)

    dice = torch.zeros(batch_size, 3)

    for batch_idx in range(batch_size):
        for class_idx in range(num_classes):
            intersection = torch.sum(label_true[batch_idx] == class_idx) * torch.sum(label_pred[batch_idx] == class_idx)
            union = torch.sum(label_true[batch_idx] == class_idx) + torch.sum(label_pred[batch_idx] == class_idx)
            
            dice[batch_idx, class_idx] = (2. * intersection + smooth) / (union + smooth)
    
    return dice


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    return dice



def one_hot_encode(label_tensor, num_classes):
    one_hot = torch.nn.functional.one_hot(label_tensor.long(), num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    return one_hot

def dice_coefficient(label_true, label_pred):
    smooth = 1e-5  
    batch_size = label_true.size(0)
    num_classes = label_true.size(1)

    dice = torch.zeros(batch_size, num_classes)

    for batch_idx in range(batch_size):
        for class_idx in range(num_classes):
            intersection = torch.sum(label_true[batch_idx, class_idx] * label_pred[batch_idx, class_idx])
            union = torch.sum(label_true[batch_idx, class_idx]) + torch.sum(label_pred[batch_idx, class_idx])
            
            dice[batch_idx, class_idx] = (2. * intersection + smooth) / (union + smooth)
    
    return dice




def calculate_accuracy(label_true, label_pred, mask):

    masked_label_true = label_true[mask]
    masked_label_pred = label_pred[mask]
    
    accuracy = torch.mean((masked_label_true == masked_label_pred).float())
    
    return accuracy








def reshape_tensor_to_2D(x):
    """
    Reshape input tensor of shape [N, C, D, H, W] or [N, C, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class  = list(x.size())[1]
    if(tensor_dim == 5):
        x_perm  = x.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        x_perm  = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    y = torch.reshape(x_perm, (-1, num_class)) 
    return y 

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model.apply(initialize_weights)
    model2 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model2.apply(initialize_weights)
    model3 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model3.apply(initialize_weights)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="val")
  

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    model.train()
  
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
         
        acc_sum = 0
        dice_sum = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            keys = sampled_batch.keys()
            for key in keys:
                print(key)
        
            i = 0
            #volume_batch, label_batch, super_label_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['super_label']
            #volume_batch, label_batch, super_label_batch = volume_batch.cuda(), label_batch.cuda(), super_label_batch.cuda()
            
            volume_batch, label_batch, full_label_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['super_label']
            volume_batch, label_batch, full_label_batch = volume_batch.cuda(), label_batch.cuda(), full_label_batch.cuda()
            #label_batch = label_batch.cpu().detach().numpy()
            #full_label_batch = full_label_batch.cpu().detach().numpy()
            print(full_label_batch)
            print(full_label_batch.max())
            print(full_label_batch.min())
            print(label_batch.max())
            print(label_batch.min())
        
            mask =(label_batch > 0) & (label_batch < 4)
            accuracy = calculate_accuracy(full_label_batch, label_batch, mask)
            accuracy = torch.mean(accuracy)
            #super_label_batch[super_label_batch < 4] = 1
            #label_batch[label_batch > 0] = 1
            #acc = accuracy(full_label_batch*mask, label_batch*mask)
            #one_hot_label_true = one_hot_encode(full_label_batch*mask, 3)
            #one_hot_label_pred = one_hot_encode(label_batch*mask, 3)
            #dice = dice_coefficient(one_hot_label_true, one_hot_label_pred)
            print(accuracy)
          
            
            acc_sum = acc_sum + accuracy
            i = i + 1
      
        print(acc_sum/(i_batch+1))
      
        


        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
