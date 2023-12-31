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
from utils import losses, metrics, ramps,feature_memory, contrastive_losses
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import test_single_volume_cct, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data, help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/superpixel_dual_contrastive', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold2', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
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
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model.train().to(device)
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
    prototype_memory = feature_memory.FeatureMemory(elements_per_class=32, n_classes=num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)
    robust_loss = losses.NoiseRobustDiceLoss(gamma = 1.5)
    adv_loss=losses.VAT2d(epi=args.magnitude)
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
            volume_batch, label_batch,super_label_batch = volume_batch.to(device), label_batch.to(device),super_label_batch.to(device)
           
            outputs,embedding,outputs_aux,embedding_aux = model(
                volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_soft_aux = torch.softmax(outputs_aux, dim=1)
            # pce loss for dual model
            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_ce2 = ce_loss(outputs_aux, label_batch[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)
            
            # MLS loss for dual model
            beta = random.random() + 1e-10
            pseudo_supervision = torch.argmax(
                (beta * outputs_soft.detach() + (1.0-beta) * outputs_soft_aux.detach()), dim=1, keepdim=False)
            loss_pse_sup = 0.5 * (dice_loss(outputs_soft, pseudo_supervision.unsqueeze(1)) + dice_loss(outputs_soft_aux, pseudo_supervision.unsqueeze(1)))
            # sNR loss for dual model
            loss_super_noise_robust = 0.5 * (dice_loss(outputs_soft, super_label_batch.unsqueeze(1)) + dice_loss(outputs_soft_aux, super_label_batch.unsqueeze(1)))
            loss_super_robust1 = robust_loss(outputs_soft,super_label_batch)
            loss_super_robust2 = robust_loss(outputs_soft_aux,super_label_batch)
            # CR loss for dual model
            labeled_features = embedding
            ### select the correct predictions and ignore the background cla
            prediction_label = torch.argmax(outputs_soft.detach() , dim=1, keepdim=False)
            mask_prediction_correctly = ((prediction_label == pseudo_supervision).float() * (prediction_label > 0).float()).bool()
            # Apply the filter mask to the features and its labels
            labeled_features = labeled_features.permute(0, 2, 3, 1)
            labels_correct = pseudo_supervision[mask_prediction_correctly]
            labeled_features_correct = labeled_features[mask_prediction_correctly, ...]
            # get projected features
            with torch.no_grad():
                model.eval()
                proj_labeled_features_correct = model.projection_head(labeled_features_correct)
                model.train()
            # updated memory bank
            prototype_memory.add_features_from_sample_learned(model, proj_labeled_features_correct, labels_correct)
            labeled_features_all = labeled_features.reshape(-1, labeled_features.size()[-1])
            labeled_labels = pseudo_supervision.reshape(-1)
            # get predicted features
            proj_labeled_features_all = model.projection_head(labeled_features_all)
            pred_labeled_features_all = model.prediction_head(proj_labeled_features_all)
            loss_contr_labeled = contrastive_losses.contrastive_class_to_class_learned_memory(model, pred_labeled_features_all, labeled_labels, num_classes, prototype_memory.memory)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            
            labeled_features_aux = embedding_aux
            prediction_label_aux = torch.argmax(outputs_soft_aux.detach() , dim=1, keepdim=False)
            ### select the correct predictions and ignore the background cla
            mask_prediction_correctly_aux = ((prediction_label_aux == pseudo_supervision).float() * (prediction_label_aux > 0).float()).bool()
            # Apply the filter mask to the features and its labels
            labeled_features_aux = labeled_features_aux.permute(0, 2, 3, 1)
            labels_correct_aux = pseudo_supervision[mask_prediction_correctly_aux]
            labeled_features_correct_aux = labeled_features_aux[mask_prediction_correctly_aux, ...]
            # get projected features
            with torch.no_grad():
                model.eval()
                proj_labeled_features_correct_aux = model.projection_head(labeled_features_correct_aux)
                model.train()
            # updated memory bank
            prototype_memory.add_features_from_sample_learned(model, proj_labeled_features_correct_aux, labels_correct_aux)
            labeled_features_all_aux = labeled_features_aux.reshape(-1, labeled_features_aux.size()[-1])
            labeled_labels_aux = pseudo_supervision.reshape(-1)
            # get predicted features
            proj_labeled_features_all_aux = model.projection_head(labeled_features_all_aux)
            pred_labeled_features_all_aux = model.prediction_head(proj_labeled_features_all_aux)
            loss_contr_labeled_aux = contrastive_losses.contrastive_class_to_class_learned_memory(model, pred_labeled_features_all_aux, labeled_labels_aux, num_classes, prototype_memory.memory)
            loss_contr = loss_contr_labeled+loss_contr_labeled_aux
            loss = loss_ce + 0.5 * loss_pse_sup + 0.005 * loss_super_noise_robust + 0.1 * consistency_weight * loss_contr
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
                'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup: %f, alpha: %f,loss_super_noise_robust: %f, loss_contr: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_pse_sup.item(), alpha,loss_super_noise_robust.item(),loss_contr.item()))

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
                    best_list = metric_list
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    best_metrics = np.mean(best_list, axis=0)
                 
                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f dc : %f jc : %f asd : %f assd : %f precision : %f ravd : %f ' % (iter_num, best_metrics[0], best_metrics[1],best_metrics[2],best_metrics[3],best_metrics[4],best_metrics[5],best_metrics[6],best_metrics[7]))
                    
                print(best_list)
                model.train()

            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            if iter_num % 30000 == 0:
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