import argparse
import logging
import os
import random
import shutil
import sys
import time

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
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import test_single_volume_cct, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/pCE_SPS', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_cct_zm', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
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
def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length
def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

def train(args, snapshot_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")

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
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
     
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch,super_label_batch= sampled_batch['image'], sampled_batch['label'],sampled_batch['super_label']
            volume_batch, label_batch,super_label_batch = volume_batch.cuda(), label_batch.cuda(),super_label_batch.cuda()
           
            outputs, outputs_aux1 = model(volume_batch)
            print(volume_batch.shape)
            outputs_soft1 = torch.softmax(outputs, dim=1)
            outputs_soft2 = torch.softmax(outputs_aux1, dim=1)
            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
            #loss_ce1_super = ce_loss(outputs, super_label_batch[:].long())
            #loss_ce2_super = ce_loss(outputs_aux1, super_label_batch[:].long())
            #loss_em1 = entropy_loss(outputs_soft1)
            #loss_em2 = entropy_loss(outputs_soft2)
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)
            #loss_ce = loss_ce1
            #loss_em = 0.5 * (loss_em1 + loss_em2)
            #print(loss_em)
            #mask = torch.zeros([volume_batch.shape[0],4,volume_batch.shape[2], volume_batch.shape[3]]).cuda()
            #mask1 = mask
            #mask2 = mask
            #print(std_map1.max())
            #print(std_map1.min())
            #print(std_map1.mean())
            #mask1[std_map1<1] = 1.0
            #mask2[std_map2<1] = 1.0
            #mask = mask1*mask2
            super_label_one_hot = one_hot_encoder(super_label_batch.unsqueeze(1),4)
            predict = reshape_tensor_to_2D(super_label_one_hot)
            soft_y1  = reshape_tensor_to_2D(outputs_soft1) 
            numerator1 = torch.abs(predict - soft_y1)
            numerator1 = torch.pow(numerator1, 1.5)
            denominator1 = predict + soft_y1 
            numer_sum1 = torch.sum(numerator1,  dim = 0)
            denom_sum1 = torch.sum(denominator1,  dim = 0)
            loss_vector1 = numer_sum1 / (denom_sum1 + 1e-5)
            loss_noise_robust1 = torch.mean(loss_vector1)
            soft_y2  = reshape_tensor_to_2D(outputs_soft2) 
            numerator2 = torch.abs(predict - soft_y2)
            numerator2 = torch.pow(numerator2, 1.5)
            denominator2 = predict + soft_y2 
            numer_sum2 = torch.sum(numerator2,  dim = 0)
            denom_sum2 = torch.sum(denominator2,  dim = 0)
            loss_vector2 = numer_sum2 / (denom_sum2 + 1e-5)
            loss_noise_robust2 = torch.mean(loss_vector2) 
            loss_super_noise_robust = 0.5 * ( loss_noise_robust1 + loss_noise_robust2 )
            
            soft_y3  = reshape_tensor_to_2D(0.5 * ( outputs_soft1 + outputs_soft2) )
            numerator3 = torch.abs(predict - soft_y3)
            numerator3 = torch.pow(numerator3, 1.5)
            denominator3 = predict + soft_y3 
            numer_sum3 = torch.sum(numerator3,  dim = 0)
            denom_sum3 = torch.sum(denominator3,  dim = 0)
            loss_vector3 = numer_sum3 / (denom_sum3 + 1e-5)
            loss_noise_robust3 = torch.mean(loss_vector3)
            #loss_super_noise_robust = loss_noise_robust3
            
            
            beta = random.random() + 1e-10
            
            pseudo_supervision = torch.argmax(
                (beta * outputs_soft1.detach() + (1.0-beta) * outputs_soft2.detach()), dim=1, keepdim=False)
            loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(1)) + dice_loss(outputs_soft2, pseudo_supervision.unsqueeze(1)))
            
            loss = loss_ce + 0.5 * loss_pse_sup 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup: %f, alpha: %f,loss_super_noise_robust: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_pse_sup.item(), alpha,loss_super_noise_robust.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_cct(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                dc = np.mean(metric_list, axis=0)[2]
                jc = np.mean(metric_list, axis=0)[3]
                asd = np.mean(metric_list, axis=0)[4]
                assd = np.mean(metric_list, axis=0)[5]
                precision = np.mean(metric_list, axis=0)[6]
                ravd = np.mean(metric_list, axis=0)[7]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f dc : %f jc : %f asd : %f assd : %f precision : %f ravd : %f ' % (iter_num, performance, mean_hd95,dc,jc,asd,assd,precision,ravd))
                model.train()

            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
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