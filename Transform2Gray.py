# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 11:52:48 2016

@author: Mukrow
"""
import pickle as pkl
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


TEST_PATH = r"E:\Distracted Driver Dataset\test"
CLASS_PATH_c0 = r"E:\Distracted Driver Dataset\train\c0"
CLASS_PATH_c1 = r"E:\Distracted Driver Dataset\train\c1"
CLASS_PATH_c2 = r"E:\Distracted Driver Dataset\train\c2"
CLASS_PATH_c3 = r"E:\Distracted Driver Dataset\train\c3"
CLASS_PATH_c4 = r"E:\Distracted Driver Dataset\train\c4"
CLASS_PATH_c5 = r"E:\Distracted Driver Dataset\train\c5"
CLASS_PATH_c6 = r"E:\Distracted Driver Dataset\train\c6"
CLASS_PATH_c7 = r"E:\Distracted Driver Dataset\train\c7"
CLASS_PATH_c8 = r"E:\Distracted Driver Dataset\train\c8"
CLASS_PATH_c9 = r"E:\Distracted Driver Dataset\train\c9"
ALL_FILES_PATH = [CLASS_PATH_c0,CLASS_PATH_c1,CLASS_PATH_c2,CLASS_PATH_c3,CLASS_PATH_c4,CLASS_PATH_c5,CLASS_PATH_c6,CLASS_PATH_c7,CLASS_PATH_c8,CLASS_PATH_c9]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_images_from_folder(folder):
    images = []
    filenames = os.listdir(folder)
    i =1
    n = len(filenames)
    for filename in filenames:
#       read image
        img = cv2.imread(os.path.join(folder,filename))
#        jaga" file bukan gambar
        if img is not None:
#           parameter buat rezize jd gambar dengan lebar 100 piksel        
            r = 100.0 / img.shape[1]
            dim = (100, int(img.shape[0]*r))
#           rezize gambar
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
#            conver jd grayscale
            img = rgb2gray(img)
#           diratakan buat jd parameter
            img = img.flatten()
            images.append(img)
#            print str(i) + '/' + str(n)
            i+=1
    full = np.c_[np.array(filenames),np.array(images)]
    return full, n

def encode_kelas(n, n_kelas, kelas_ke):
    kelas = np.zeros((n, n_kelas))
    for i in range(n):
        kelas[i,kelas_ke]=1
    return kelas
        
    
def buat_dataset_n_kelas():
    data, n_dat = load_images_from_folder(ALL_FILES_PATH[0])
    n_all=[n_dat]
    kelas = encode_kelas(n_dat,10,0)
    for i in range(1,len(ALL_FILES_PATH)):
        tmp, n_tmp = load_images_from_folder(ALL_FILES_PATH[i])
        tmp_kelas = encode_kelas(n_tmp, 10, i)
        data = np.append(data, tmp, axis=0)
        kelas = np.append(kelas, tmp_kelas, axis=0)
        print "kelas ke-%d" %(i)
#        print "jumlah data = %d" %(n_tmp)
        n_all.append(n_tmp)
#        input()
    del tmp
    print "total data = %d" %(sum(n_all))
    return data, kelas
        
x, y = buat_dataset_n_kelas()        
dat = np.c_[x,y]
np.savetxt('trainset_distracted_driver.csv',x,delimiter=',', fmt='%s')
np.savetxt('kelas_train_distracted_driver.csv',y,delimiter=',', fmt='%d')
np.savetxt('dataset_full.csv', dat, delimiter=',',fmt='%s')
print 'alles hat gemacht'

