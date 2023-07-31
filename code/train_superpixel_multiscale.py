
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
from dataloaders.dataset_super import BaseDataSets, RandomGenerator2
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds, test_single_volume_cct

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/pCE_Random_Walker', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size1', type=list,  default=[192,192],
                    help='patch size of network input')
parser.add_argument('--patch_size2', type=list,  default=[128,128],
                    help='patch size of network input')
parser.add_argument('--patch_size3', type=list,  default=[256,256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
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
def add_gaussian_noise(img_batch, mean=0.0, std=0.1):
    noise = torch.randn_like(img_batch) * std + mean
    noisy_imgs = img_batch + noise
    noisy_imgs = torch.clamp(noisy_imgs, 0, 1)
    return noisy_imgs
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def train(args, snapshot_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator2(args.patch_size1,args.patch_size2,args.patch_size3)
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
            volume_batch1, label_batch1, super_label_batch1 = sampled_batch['image'], sampled_batch['label'], sampled_batch['super_label']
            volume_batch2, label_batch2, super_label_batch2 = sampled_batch['image2'], sampled_batch['label2'], sampled_batch['super_label2']
            volume_batch3, label_batch3, super_label_batch3 = sampled_batch['image3'], sampled_batch['label3'], sampled_batch['super_label3']
            
            volume_batch1, label_batch1, super_label_batch1 = volume_batch1.cuda(), label_batch1.cuda(), super_label_batch1.cuda()
            volume_batch2, label_batch2, super_label_batch2 = volume_batch2.cuda(), label_batch2.cuda(), super_label_batch2.cuda()
            volume_batch3, label_batch3, super_label_batch3 = volume_batch3.cuda(), label_batch3.cuda(), super_label_batch3.cuda()
            volume_batch1 = add_gaussian_noise(volume_batch1, mean=0.0, std=0.1)  
            volume_batch2 = add_gaussian_noise(volume_batch2, mean=0.0, std=0.1)  
            volume_batch3 = add_gaussian_noise(volume_batch3, mean=0.0, std=0.1)  
            
            
            #volume_batch1 = volume_batch1 + torch.clamp(torch.randn_like(volume_batch1) * 0.1, -0.2, 0.2)
            #volume_batch2 = volume_batch2 + torch.clamp(torch.randn_like(volume_batch2) * 0.1, -0.2, 0.2)
            #volume_batch3 = volume_batch3 + torch.clamp(torch.randn_like(volume_batch3) * 0.1, -0.2, 0.2)

            outputs1 = model(volume_batch1)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs2  = model(volume_batch2)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            outputs3  = model(volume_batch3)
            outputs_soft3 = torch.softmax(outputs3, dim=1)
            
            
            outputs_soft1_ = F.interpolate(outputs_soft1, size=(256,256), mode='bilinear', align_corners=False)
            outputs_soft2_ = F.interpolate(outputs_soft2, size=(256,256), mode='bilinear', align_corners=False)
            pseudo_label = torch.argmax(
                (0.25 * outputs_soft1_.detach() + 0.25 * outputs_soft2_.detach() + 0.5 * outputs_soft3.detach()), dim=1, keepdim=False)
            avg_outpus_soft = 0.25 * outputs_soft1_ + 0.25 * outputs_soft2_ + 0.5 * outputs_soft3
            #pseudo_supervision = pseudo_label
            loss_pse_sup = 0.25 * (dice_loss(outputs_soft1_, pseudo_label.unsqueeze(1)) + dice_loss(outputs_soft2_, pseudo_label.unsqueeze(1))) + 0.5 * (dice_loss(outputs_soft3, pseudo_label.unsqueeze(1)))
            beta = random.random() + 1e-10
            #pseudo_supervision = torch.argmax(
                #(beta * outputs_soft3.detach() + (1.0-beta) * outputs_aux_soft3.detach()), dim=1, keepdim=False)
            #loss_pse_sup = 0.5 * (dice_loss(outputs_soft3, pseudo_supervision.unsqueeze(1)) + dice_loss(outputs_aux_soft3, pseudo_supervision.unsqueeze(1)))
            loss_ce1 =  ce_loss(outputs3, label_batch3[:].long())
            loss_ce2 =  ce_loss(outputs1, label_batch1[:].long())
            loss_ce3 =  ce_loss(outputs2, label_batch2[:].long())
            loss_ce = 0.5 * loss_ce1 + 0.25 * loss_ce2 + 0.25 * loss_ce3
            ent_loss = losses.entropy_loss(outputs_soft3, C=4)
            
            dice_loss_1 = dice_loss(outputs_soft3,label_batch3.unsqueeze(1))
            super_label_one_hot1 = one_hot_encoder(super_label_batch1.unsqueeze(1),4)
            predict1 = reshape_tensor_to_2D(super_label_one_hot1)
            soft_y1  = reshape_tensor_to_2D(outputs_soft1) 
            numerator1 = torch.abs(predict1 - soft_y1)
            numerator1 = torch.pow(numerator1, 1.5)
            denominator1 = predict1 + soft_y1
            numer_sum1 = torch.sum(numerator1,  dim = 0)
            denom_sum1 = torch.sum(denominator1,  dim = 0)
            loss_vector1 = numer_sum1 / (denom_sum1 + 1e-5)
            loss_noise_robust1 = torch.mean(loss_vector1) 
            
            
            super_label_one_hot2 = one_hot_encoder(super_label_batch2.unsqueeze(1),4)
            predict2 = reshape_tensor_to_2D(super_label_one_hot2)
            soft_y2  = reshape_tensor_to_2D(outputs_soft2) 
            numerator2 = torch.abs(predict2 - soft_y2)
            numerator2 = torch.pow(numerator2, 1.5)
            denominator2 = predict2 + soft_y2
            numer_sum2 = torch.sum(numerator2,  dim = 0)
            denom_sum2 = torch.sum(denominator2,  dim = 0)
            loss_vector2 = numer_sum2 / (denom_sum2 + 1e-5)
            loss_noise_robust2 = torch.mean(loss_vector2) 
            
            
            super_label_one_hot3 = one_hot_encoder(super_label_batch3.unsqueeze(1),4)
            predict3 = reshape_tensor_to_2D(super_label_one_hot3)
            soft_y3  = reshape_tensor_to_2D(outputs_soft3) 
            numerator3 = torch.abs(predict3 - soft_y3)
            numerator3 = torch.pow(numerator3, 1.5)
            denominator3 = predict3 + soft_y3
            numer_sum3 = torch.sum(numerator3,  dim = 0)
            denom_sum3 = torch.sum(denominator3,  dim = 0)
            loss_vector3 = numer_sum3 / (denom_sum3 + 1e-5)
            loss_noise_robust3 = torch.mean(loss_vector3) 
            
            
            

            
            loss_noise_robust = 0.3 * loss_noise_robust1 + 0.3 * loss_noise_robust2 + 0.4 * loss_noise_robust3
            
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            loss = 1.0 * loss_ce + consistency_weight * loss_noise_robust + consistency_weight * loss_pse_sup
            # loss = loss_ce
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
                'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup: %f, loss_noise_robust: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_pse_sup.item(),loss_noise_robust.item() ))

            if iter_num % 20 == 0:
                image = volume_batch3[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs3, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch3[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 20 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
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
