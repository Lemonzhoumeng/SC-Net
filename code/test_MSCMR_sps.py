import argparse
import os
import re
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from skimage import transform
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from torch.utils.data import DataLoader
# from networks.efficientunet import UNet
from networks.net_factory import net_factory
import data_MSCMR 
from data_MSCMR import build
from val_2D import test_single_volume_cct, test_single_volume_ds
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/zm/WSL4MIS/MSCMR_dataset/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MSCMR/pCE_SPS', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')
parser.add_argument('--dataset', default='MSCMR_dataset', type=str,
                        help='multi-sequence CMR segmentation dataset')                      
tasks = {
        'MR': {'lab_values': [0, 1, 2, 3, 4, 5], 'out_channels': 4}
        }
parser.add_argument('--tasks', default=tasks, type=dict)
args = parser.parse_args()

def read_image(img_path):
    img_dat = load_nii(img_path)
    img = img_dat[0]
    pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
    target_resolution = (1.36719, 1.36719)
    scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
    img = img.astype(np.float32)
    return [(img-img.mean())/img.std(), scale_vector]
def makefolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False
def read_label(lab_path):
    lab_dat = load_nii(lab_path)
    lab = lab_dat[0]
    pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
    target_resolution = (1.36719, 1.36719)
    scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        # cla = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
    return [lab, scale_vector]
    
def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header
def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)
def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    return dice


def test_single_volume(case,volume_batch,label_batch, net, test_save_path, FLAGS):
    image = volume_batch
    label = label_batch
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_aux1, out_aux2 = net(input)
            out_aux1_soft = torch.softmax(out_aux1, dim=1)
            out_aux2_soft = torch.softmax(out_aux2, dim=1)
            out = torch.argmax(out_aux1_soft, dim=1).squeeze(0)
            # out = torch.argmax((out_aux1_soft+out_aux2_soft)/2.0, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    org_img_path = os.path.join("/home/zm/WSL4MIS/MSCMR_dataset/test/images/",str(case))
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(
        FLAGS.exp, FLAGS.sup_type)
    test_save_path = "../model/{}/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, 'iter_48000.pth')

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    print('Building test dataset...')
    test_folder = "/home/zm/WSL4MIS/MSCMR_dataset/test/images/"
    label_folder = "/home/zm/WSL4MIS/MSCMR_dataset/test/labels/"
    output_folder = "/home/zm/WSL4MIS/model/MSCMR/pCE_SPS/scribble/unet_cct_predictions/"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    makefolder(output_folder)
    target_resolution = (1.36719, 1.36719)
    test_files = sorted(os.listdir(test_folder))
    label_files = sorted(os.listdir(label_folder))
    assert len(test_files) == len(label_files)
    # read_image
    for file_index in range(len(test_files)):
        test_file = test_files[file_index] 
        label_file = label_files[file_index]
        file_mask = os.path.join(label_folder, label_file)
        mask_dat = load_nii(file_mask)
        mask = mask_dat[0]
        img_path = os.path.join(test_folder, test_file)
        img_dat = load_nii(img_path)
        img = img_dat[0].copy()

        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])

        img = img.astype(np.float32)
        img = np.divide((img - np.mean(img)), np.std(img))
        slice_rescaleds = []
        for slice_index in range(img.shape[2]):
            img_slice = np.squeeze(img[:,:,slice_index])
            slice_rescaled = transform.rescale(img_slice,
                                            scale_vector,
                                            order=1,
                                            preserve_range=True,
                                            multichannel=False,
                                            anti_aliasing=True,
                                            mode='constant')
            slice_rescaleds.append(slice_rescaled)
        img = np.stack(slice_rescaleds, axis=2)
        prediction = []
        for slice_index in range(img.shape[2]):
            
            img_slice = img[:,:,slice_index]
            nx = 256
            ny = 256
            x, y = img_slice.shape
            x_s = (x - nx) // 2
            y_s = (y - ny) // 2
            x_c = (nx - x) // 2
            y_c = (ny - y) // 2
            # Crop section of image for prediction
            if x > nx and y > ny:
                slice_cropped = img_slice[x_s:x_s+nx, y_s:y_s+ny]
            else:
                slice_cropped = np.zeros((nx,ny))
                if x <= nx and y > ny:
                    slice_cropped[x_c:x_c+ x, :] = img_slice[:,y_s:y_s + ny]
                elif x > nx and y <= ny:
                    slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
                else:
                    slice_cropped[x_c:x_c+x, y_c:y_c + y] = img_slice[:, :]
            
            img_slice = slice_cropped
            img_slice = np.divide((slice_cropped - np.mean(slice_cropped)), np.std(slice_cropped))
            img_slice = np.reshape(img_slice, (1,1,nx,ny))

            img_slice = torch.from_numpy(img_slice)
            img_slice = img_slice.cuda()
            img_slice = img_slice.float()
            net.eval()
         
            with torch.no_grad():
                out_aux1, out_aux2 = net(img_slice)
                out_aux1_soft = torch.softmax(out_aux1, dim=1)
                out_aux2_soft = torch.softmax(out_aux2, dim=1)
                out = torch.argmax(out_aux1_soft, dim=1).squeeze(0)
            # out = torch.argmax((out_aux1_soft+out_aux2_soft)/2.0, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(out, (x / 256, y / 256), order=0)
                pred1 = pred.tolist()
                prediction.append(pred1)
        predictions = np.array(prediction)
        dir_pred = os.path.join(output_folder, "predictions")
        makefolder(dir_pred)
        out_file_name = os.path.join(dir_pred, label_file)
        out_affine = mask_dat[1]
        out_header = mask_dat[2]
        save_nii(out_file_name, predictions, out_affine, out_header)
        dir_gt = os.path.join(output_folder, "masks")
        makefolder(dir_gt)
        mask_file_name = os.path.join(dir_gt, label_file)
        save_nii(mask_file_name, mask_dat[0], out_affine, out_header)
    
    db_test = build(image_set='test', args=args)
    testloader = DataLoader(db_test,batch_size=1,shuffle=False,num_workers=0)
    num_classes = 4
    print('Number of val images: {}'.format(len(testloader)))
    metric_list = 0.0
    print(img_paths)
    metric_dice1 = []
    metric_dice2 = []
    metric_dice3 = []
    for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
        org_img_path = os.path.join("/home/zm/WSL4MIS/MSCMR_dataset/test/images/",str(img_path))
        img = read_image(org_img_path)
        org_label_path = os.path.join("/home/zm/WSL4MIS/MSCMR_dataset/test/labels/",str(lab_path))
        label = read_label(org_label_path)
        image = img[0]
        label = label[0]
        prediction = np.zeros_like(label)
        for ind in range(image.shape[2]):
            slice = image[:,:,3]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (256 / x, 256 / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out_aux1, out_aux2 = net(input)
                out_aux1_soft = torch.softmax(out_aux1, dim=1)
                out_aux2_soft = torch.softmax(out_aux2, dim=1)
                out = torch.argmax(out_aux1_soft, dim=1).squeeze(0)
            # out = torch.argmax((out_aux1_soft+out_aux2_soft)/2.0, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(out, (x / 256, y / 256), order=0)
                prediction[ind] = pred
        print(input.shape)
        print(pred.shape)
        print(label.max())
        print(prediction.shape)
        org_img_itk = image
        #spacing = org_img_itk.GetSpacing()
        #first_metric = calculate_metric_percase(
        #prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
        #second_metric = calculate_metric_percase(
        #prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
        #third_metric = calculate_metric_percase(
        #prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        print(org_img_itk.shape)
        #img_itk.CopyInformation(org_img_itk)
        
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.CopyInformation(org_img_itk)
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk.CopyInformation(org_img_itk)
        #sitk.WriteImage(prd_itk, test_save_path + str(img_path) + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + str(img_path) + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + str(img_path) + "_gt.nii.gz")
"""
    for i_batch, sampled_batch in enumerate(testloader):
        volume_batch, label_all_batch = sampled_batch
            #print(volume_batch.shape)
        label_batch = label_all_batch["masks"]
        volume_batch, label_batch = volume_batch.cuda().float(), label_batch.cuda().float()
        metric_i,label,prediction = test_single_volume_cct(volume_batch, label_batch, net, classes=num_classes)
        metric_list += np.array(metric_i)
        metric_dice1.append(metric_i[0])
        metric_dice2.append(metric_i[1])
        metric_dice3.append(metric_i[2])
        print(prediction.mean())
        
    metric_list = metric_list / len(db_test)
    std = [np.std(metric_dice1), np.std(metric_dice2), np.std(metric_dice3)]
    print(metric_list)
    print(std)
"""
        
if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
