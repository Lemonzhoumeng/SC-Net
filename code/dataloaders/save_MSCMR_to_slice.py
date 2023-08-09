#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 18:52:03 2023

@author: zhoumeng
"""

# save images in slice level
import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())

# saving images in slice level


slice_num = 0
"""
mask_path = sorted(
    glob.glob('//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//TestSet//*_manual.nii.gz'))
for case in mask_path:
    label_itk = sitk.ReadImage(case)
    label = sitk.GetArrayFromImage(label_itk)

    image_path = case.replace("_manual", "")
    image_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_itk)

    image = MedicalImageDeal(image, percent=0.99).valid_img
    image = (image - image.min()) / (image.max() - image.min())
    print(image.shape)
    image = image.astype(np.float32)
    item = case.split("/")[-1].split(".")[0].replace("_gt", "")
    if image.shape != label.shape:
        print("Error")
    print(item)
    for slice_ind in range(image.shape[0]):
        f = h5py.File(
            '//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//MSCMR_test_slices//{}_slice_{}.h5'.format(item, slice_ind), 'w')
        f.create_dataset(
            'image', data=image[slice_ind], compression="gzip")
        f.create_dataset('label', data=label[slice_ind], compression="gzip")
        f.close()
        slice_num += 1
print("Converted all MSCMR volumes to 2D slices")
print("Total {} slices".format(slice_num))


mask_path = sorted(
    glob.glob('//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//val//*_manual.nii.gz'))
for case in mask_path:
    label_itk = sitk.ReadImage(case)
    label = sitk.GetArrayFromImage(label_itk)

    image_path = case.replace("_manual", "")
    image_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_itk)

    image = MedicalImageDeal(image, percent=0.99).valid_img
    image = (image - image.min()) / (image.max() - image.min())
    print(image.shape)
    image = image.astype(np.float32)
    item = case.split("/")[-1].split(".")[0].replace("_gt", "")
    if image.shape != label.shape:
        print("Error")
    print(item)
    for slice_ind in range(image.shape[0]):
        f = h5py.File(
            '//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//MSCMR_val_slices//{}_slice_{}.h5'.format(item, slice_ind), 'w')
        f.create_dataset(
            'image', data=image[slice_ind], compression="gzip")
        f.create_dataset('label', data=label[slice_ind], compression="gzip")
        f.close()
        slice_num += 1
print("Converted all MSCMR volumes to 2D slices")
print("Total {} slices".format(slice_num))



scribble_path = sorted(
    glob.glob('//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//train//*_scribble.nii.gz'))
for case in scribble_path:
    print(case)
    scribble_itk = sitk.ReadImage(case)
    scribble = sitk.GetArrayFromImage(scribble_itk)
    image_path = case.replace("_scribble", "")
    image_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_itk)

    image = MedicalImageDeal(image, percent=0.99).valid_img
    image = (image - image.min()) / (image.max() - image.min())
    print(image.shape)
    image = image.astype(np.float32)
    item = case.split("/")[-1].split(".")[0].replace("_scribble", "")
    if image.shape != scribble.shape:
        print("Error")
    print(item)
    for slice_ind in range(image.shape[0]):
        f = h5py.File(
            '//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//MSCMR_train_slices//{}_slice_{}.h5'.format(item, slice_ind), 'w')
        f.create_dataset(
            'image', data=image[slice_ind], compression="gzip")
        f.create_dataset('scribble', data=scribble[slice_ind], compression="gzip")
        f.close()
        slice_num += 1
print("Converted all MSCMR volumes to 2D slices")
print("Total {} slices".format(slice_num))
"""


# saving images in volume level


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())


slice_num = 0
"""
scribble_path = sorted(
    glob.glob("//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//train//*_scribble.nii.gz"))
for case in scribble_path:

    image_path = case.replace("_scribble", "")
    image_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_itk)
    scribble_itk = sitk.ReadImage(case)
    scribble = sitk.GetArrayFromImage(scribble_itk)
  
    image = MedicalImageDeal(image, percent=0.99).valid_img
    image = (image - image.min()) / (image.max() - image.min())
    print(image.shape)
    image = image.astype(np.float32)
    item = case.split("/")[-1].split(".")[0].replace("_gt", "")
    if image.shape != scribble.shape:
        print("Error")
    print(item)
    f = h5py.File(
        '//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//MSCMR_train_volumes//{}.h5'.format(item), 'w')
    f.create_dataset(
        'image', data=image, compression="gzip")
    f.create_dataset('scribble', data=scribble, compression="gzip")
    #f.create_dataset('scribble', data=scribble, compression="gzip")
    f.close()
    slice_num += 1
print("Converted all MSCMR volumes to 2D slices")
print("Total {} slices".format(slice_num))

"""


label_path = sorted(
    glob.glob("//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//val//*_manual.nii.gz"))
for case in label_path:

    image_path = case.replace("_manual", "")
    image_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_itk)
    label_itk = sitk.ReadImage(case)
    label = sitk.GetArrayFromImage(label_itk)
  
    image = MedicalImageDeal(image, percent=0.99).valid_img
    image = (image - image.min()) / (image.max() - image.min())
    print(image.shape)
    image = image.astype(np.float32)
    item = case.split("/")[-1].split(".")[0].replace("_gt", "")
    if image.shape != label.shape:
        print("Error")
    print(item)
    f = h5py.File(
        '//Users//zhoumeng//Downloads//CycleMix-main//MSCMR_dataset//MSCMR_val_volumes//{}.h5'.format(item), 'w')
    f.create_dataset(
        'image', data=image, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    #f.create_dataset('scribble', data=scribble, compression="gzip")
    f.close()
    slice_num += 1
print("Converted all MSCMR volumes to 2D slices")
print("Total {} slices".format(slice_num))





























