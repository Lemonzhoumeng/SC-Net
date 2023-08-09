#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 20:29:41 2023

@author: zhoumeng
"""

import math
from skimage import io, color
import numpy as np
from tqdm import trange
import glob
import os
import cv2
import h5py
import numpy as np
import SimpleITK as sitk
import matplotlib
from PIL import Image





class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        :param path:
        :param lab_arr:
        :return:
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename, K, M):
        self.K = K
        self.M = M

        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)
    
    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S
       
    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient
    def assignment(self):
        for i in range(len(self.clusters)):
            cluster = self.clusters[i]
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            _h = int(sum_h / number)
            _w = int(sum_w / number)
            cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)
    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        for i in range(10):
            self.assignment()
            self.update_cluster()
            name = 'test_M{m}_K{k}_loop{loop}.png'.format(loop=i, m=self.M, k=self.K)
            self.save_current_image(name)
        return self.clusters
    def generate_superpixel(self):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        return image_arr

if __name__ == '__main__':
    all_slices = os.listdir('../data/ACDC/ACDC_training_slices')
    for k in range(len(all_slices)):
        case = all_slices[k]
        h5f = h5py.File('../data/ACDC/ACDC_training_slices/{}'.format(case), 'r')
        image = h5f['image'][:]
        label_original = h5f['label'][:]
    
        data = image*255.0
        print(data.shape)
        w,h = data.shape
        label_show = np.zeros([w,h,3])
        scribble = h5f['scribble'][:]
        outimg = Image.fromarray(data)
        outimg = outimg.convert('L')
        outimg.save('test.bmp')
        src = cv2.imread("test.bmp", 0)
        src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("test.jpg", src_RGB)

        p = SLICProcessor('test.jpg', 200, 15)
        clusters = p.iterate_10times()
        image_superpixel = p.generate_superpixel()
        scribble2 = scribble.copy()
        scribble_super_show = np.zeros([w,h,3])
        for i in range(len(clusters)):
            num0 = 0
            num1 = 0
            num2 = 0
            num3 = 0
            num4 = 0
            number = []
            for j in range(len(clusters[i].pixels)):
                m,n = ((clusters[i].pixels))[j]
                if (scribble2[m,n] == 0):
                    num0+=1
                elif (scribble2[m,n] == 1):
                    num1+=1
                elif (scribble2[m,n] == 2):
                    num2+=1
                elif (scribble2[m,n] == 3):
                    num3+=1  
                else:
                    num4+=1
            number = [num0,num1,num2,num3]
            if ((num0>0) or (num1>0) or (num2>0) or (num3>0)): 
                if sum(i>0 for i in number) >1 :
                    label =4
                else:
                    label = number.index(max(number))
            else :
                label = 4
            for j in range(len(clusters[i].pixels)):
                m,n = ((clusters[i].pixels))[j]
                scribble2[m][n] = label
        h5f.close()
        f = h5py.File(
             '../data/ACDC_add_Super_M200_casuper/ACDC_training_slices/{}'.format(case), 'w')
        
        f.create_dataset(
             'super_scribble', data=scribble2, compression="gzip")
        f.create_dataset(
            'image', data=image, compression="gzip")
        f.create_dataset('scribble', data=scribble, compression="gzip") 
        f.create_dataset('label', data=label_original, compression="gzip")
        f.create_dataset('image_superpixel', data=image_superpixel, compression="gzip")
        # f.create_dataset('scribble', data=scribble[slice_ind], compression="gzip")
        # f.close()
        # slice_num += 1
        f.close()
    
        h5f1 = h5py.File("../data/ACDC/ACDC_test/ACDC_training_slices//{}".format(case), 'r')
        image = h5f1['image'][:]
        label_original = h5f1['label'][:]
        scribble = h5f1['scribble'][:]
        super_scribble = h5f1['super_scribble']
        a = super_scribble - scribble
        data = image*255.0
        w,h = data.shape
        label_show = np.zeros([w,h,3])
        outimg = Image.fromarray(data)
        outimg = outimg.convert('L')
        outimg.save('test.bmp')
        src = cv2.imread("test.bmp", 0)
        src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("test.jpg", src_RGB)
        scribble_super_show = np.zeros([w,h,3])
        scribble_original_show = np.zeros([w,h,3])
        print(super_scribble)
        for i in range(w):
            for j in range(h):
                if (super_scribble[i][j] == 0):
                    scribble_super_show[i][j][:] = [0,0,0]
                elif (super_scribble[i][j] == 1):
                    scribble_super_show[i][j][:] = [0,0,255]
                elif (super_scribble[i][j] == 2):
                    scribble_super_show[i][j][:] = [0,255,0]
                elif (super_scribble[i][j] == 3):
                    scribble_super_show[i][j][:]= [255,0,0]
                else:
                    scribble_super_show[i][j][:] = [200,200,0] 
        cv2.imwrite("test_super_scribble.jpg", scribble_super_show)
    
        for i in range(w):
            for j in range(h):
                if (scribble[i][j] == 0):
                    scribble_original_show[i][j][:] = [0,0,0]
                elif (scribble[i][j] == 1):
                    scribble_original_show[i][j][:] = [0,0,255]
                elif (scribble[i][j] == 2):
                    scribble_original_show[i][j][:] = [0,255,0]
                elif (scribble[i][j] == 3):
                    scribble_original_show[i][j][:]= [255,0,0]
                else:
                    scribble_original_show[i][j][:] = [200,200,0] 
        cv2.imwrite("test_original_scribble.jpg", scribble_original_show)
    
    

    p = SLICProcessor('test.jpg', 500, 5)
    clusters = p.iterate_10times()
    scribble250 = scribble.copy()
    scribble_super_show = np.zeros([w,h,3])
    scribble_original_show = np.zeros([w,h,3])
    for i in range(len(clusters)):
        num0 = 0
        num1 = 0
        num2 = 0
        num3 = 0
        num4 = 0
        number = []
        for j in range(len(clusters[i].pixels)):
            m,n = ((clusters[i].pixels))[j]
            if (scribble[m,n] == 0):
                num0+=1
            elif (scribble[m,n] == 1):
                num1+=1
            elif (scribble[m,n] == 2):
                num2+=1
            elif (scribble[m,n] == 3):
                num3+=1  
            else:
                num4+=1
        number = [num0,num1,num2,num3]
        if ((num0>0) or (num1>0) or (num2>0) or (num3>0)): 
            if sum(i>0 for i in number) >1 :
                label =4
            else:
                label = number.index(max(number))
        else :
            label = 4
        for j in range(len(clusters[i].pixels)):
            m,n = ((clusters[i].pixels))[j]
            scribble250[m][n] = label
    scribble_multi = scribble250 
    for i in range(w):
        for j in range(h):
            if (scribble_multi[i][j] == 0):
                scribble_super_show[i][j][:] = [0,0,0]
            elif (scribble_multi[i][j] == 1):
                scribble_super_show[i][j][:] = [0,0,255]
            elif (scribble_multi[i][j] == 2):
                scribble_super_show[i][j][:] = [0,255,0]
            elif (scribble_multi[i][j] == 3):
                scribble_super_show[i][j][:]= [255,0,0]
            else:
                scribble_super_show[i][j][:] = [200,200,0] 
    cv2.imwrite("test_super__multi_scribble.jpg", scribble_super_show)
    
    
    for i in range(w):
        for j in range(h):
            if (scribble2[i][j] == 0):
                scribble_super_show[i][j][:] = [0,0,0]
            elif (scribble2[i][j] == 1):
                scribble_super_show[i][j][:] = [0,0,255]
            elif (scribble2[i][j] == 2):
                scribble_super_show[i][j][:] = [0,255,0]
            elif (scribble2[i][j] == 3):
                scribble_super_show[i][j][:]= [255,0,0]
            else:
                scribble_super_show[i][j][:] = [200,200,0] 
    cv2.imwrite("test_super_scribble.jpg", scribble_super_show)
    
    for i in range(w):
        for j in range(h):
            if (scribble[i][j] == 0):
                scribble_original_show[i][j][:] = [0,0,0]
            elif (scribble[i][j] == 1):
                scribble_original_show[i][j][:] = [0,0,255]
            elif (scribble[i][j] == 2):
                scribble_original_show[i][j][:] = [0,255,0]
            elif (scribble[i][j] == 3):
                scribble_original_show[i][j][:]= [255,0,0]
            else:
                scribble_original_show[i][j][:] = [200,200,0] 
    cv2.imwrite("original_scribble.jpg", scribble_original_show)
    
    
    

    for i in range(w):
        for j in range(h):
            if (label_original[i][j] == 1):
                label_show[i][j][:] = [0,0,255]
            elif (label_original[i][j] == 2):
                label_show[i][j][:] = [0,255,0]
            elif (label_original[i][j] == 3):
                label_show[i][j][:] = [255,0,0]
            else:
                label_show[i][j][:] = [0,0,0]
    cv2.imwrite("test_label.jpg", label_show)
    
