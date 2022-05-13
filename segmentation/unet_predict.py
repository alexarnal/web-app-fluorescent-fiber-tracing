#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 2:57:34 2021

@author: Aryal007
"""
from segmentation.utils.data import fetch_loaders
from segmentation.utils.frame import Framework
import segmentation.utils.functions as fn

import yaml, pathlib, pickle, warnings, torch, matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm 
from addict import Dict
import numpy as np
import os
import matplotlib.image as mpimg
from time import time
from subprocess import check_output
from skimage.morphology import remove_small_objects, skeletonize
import imageio

def writeSVG(fileName,polys,height, width):
    f = open(fileName, "w")
    f.write(f'<svg viewBox="0 0 {width} {height}">\n')
    f.write('\t<path fill="none" stroke="black" stroke-width="1" d="\n')
    f.write(f'\t\tM 0, {height}\n')#M 0, 8516
    f.write(f'\t\tL {width}, {height}\n')
    f.write(f'\t\tL {width}, 0\n')
    f.write('\t\tL 0, 0\n\t\tz\n\t"/>\n')
    for l in polys:
        l = l.split(' ')
        l = [c.split(',') for c in l][:-1] 
        f.write('\t<path fill="none" stroke="black" stroke-width="1" d="\n')
        f.write(f'\t\tM {l[0][0]}, {l[0][1]}\n')
        for i in range(1,len(l)):
            f.write(f'\t\tL {l[i][0]}, {l[i][1]}\n')
        f.write('\t"/>\n')       
    f.write('</svg>')        
    f.close()

def verifyDims(img, slice_hw):
    new_dim = list(img.shape)
    if img.shape[0] % slice_hw[0] !=0: new_dim[0]=(img.shape[0]//slice_hw[0] + 1)*slice_hw[0]
    if img.shape[1] % slice_hw[1] !=0: new_dim[1]=(img.shape[1]//slice_hw[1] + 1)*slice_hw[1]
    if new_dim != img.shape:
        print(f'New canvas dimensions: {new_dim}')
        temp = np.zeros(new_dim)
        temp[0:img.shape[0],0:img.shape[1]] = img
        return temp
    return img

def verifyChannel(img, channel):
    temp = np.zeros((img.shape[0],img.shape[1],3))
    if len(img.shape)==3:
        if img.shape[2]==4: img = img[:,:,0:3]
        print(f'Using channel: {channel}')
        if channel == 'color':
            temp = img
        if channel == 'red':
            for i in range(3): temp[:,:,i] = img[:,:,0]
        elif channel == 'green':
            for i in range(3): temp[:,:,i] = img[:,:,1]
        elif channel == 'blue':
            for i in range(3): temp[:,:,i] = img[:,:,2]
        elif channel == 'gray':
            print('\tDetected RGB image; converting to Gray Scale')
            for i in range(3): temp[:,:,i] = np.max(img, axis=-1)
    elif len(img.shape)==2:
        print(f'Detected a Gray Scale image')
        for i in range(3): temp[:,:,i] = np.mean(img)
    return temp

def verifyRange(img):
    if np.max(img)>1:
        img = img/255.0
        return img
    return img

def O(I,F,P,S):
    return (I-F+P)//S + 1

def run(img_path, channel):
    t = time()

    conf = Dict(yaml.safe_load(open('segmentation/utils/unet_predict.yaml')))
    model_path = 'segmentation/model_bestVal.pt' 

    print(f'Loading image')  
    img = mpimg.imread(img_path)
    _img = verifyChannel(img, channel)
    _img = verifyDims(_img, conf["window_size"])
    _img = verifyRange(_img)
    print(np.max(_img))

    print(f'Creating u-net model instance')    
    frame = Framework(
        model_opts=conf.model_opts,
        device=conf.device
    )

    print(f'Loading model weights')
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    frame.load_state_dict(state_dict)
    filename = img_path.split(".")[0]
    x = np.expand_dims(_img, axis=0)
    x = torch.from_numpy(x).float()

    print(f'Spliting image & predicting')
    crop = conf["window_size"][0]//4 #for stitching purposes
    row_stride=conf.window_size[0]-(2*crop)
    col_stride=conf.window_size[1]-(2*crop)
    out_height, out_width = O(_img.shape[0],conf["window_size"][0],0,row_stride), O(_img.shape[1],conf["window_size"][0],0,col_stride)
    new_height = row_stride * (out_height-1) + conf["window_size"][0]
    new_width = col_stride * (out_width-1) + conf["window_size"][1]
    y = np.zeros((new_height,new_width))
    for i in range(out_height):
        row = i*row_stride 
        for j in range(out_width):
            col = j*col_stride
            current_slice = x[:, row:row+conf["window_size"][0], col:col+conf["window_size"][1], :]
            if current_slice.shape[1] != conf.window_size[0] or current_slice.shape[2] != conf.window_size[1]:
                temp = np.zeros((1, conf.window_size[0], conf.window_size[1], x.shape[3]))
                temp[:, :current_slice.shape[1], :current_slice.shape[2], :] =  current_slice
                current_slice = torch.from_numpy(temp).float()
            prediction = frame.infer(current_slice)
            prediction = np.asarray(prediction.cpu()).squeeze()[:,:,1]
            if i == 0 and j==0:
                y[row:row+conf.window_size[0],col:col+conf.window_size[1]]= prediction
            elif i == 0 and j!=0:
                y[row:row+conf.window_size[0],col+crop:col+conf.window_size[1]]= prediction[:,crop:] 
            elif i != 0 and j==0:
                y[row+crop:row+conf.window_size[0],col:col+conf.window_size[1]]= prediction[crop:,:] 
            elif i != 0 and j!=0:
                y[row+crop:row+conf.window_size[0],col+crop:col+conf.window_size[1]]= prediction[crop:,crop:] 
    
    print(f'Skeletonizing Model Output')
    y = y[0:img.shape[0],0:img.shape[1]]
    y = y > 0.5
    y = remove_small_objects(y, 9,connectivity=8).astype(int)
    skeleton = skeletonize(y).astype('uint8')*255
    imageio.imwrite(filename+'_skl.png', skeleton) #saving skeleton for vectorization c program
    
    print(f'Vectorizing Skeleton')
    polylines = check_output(["segmentation/a.out", filename+'_skl.png', "to", "a"]) #running c program as a subprocess
    os.remove(filename+'_skl.png')
    p = polylines.decode("utf-8") 
    p = p.split('\n')[:-1]
    writeSVG(filename+'.svg',p,y.shape[0], y.shape[1])
    plt.imsave(filename+'_pred.png', y, cmap='gray', vmin=0, vmax=1)
    
    t = time()-t
    print(f'Finished segmenting and tracing {filename} w dims {y.shape} in {t} sec\n') 