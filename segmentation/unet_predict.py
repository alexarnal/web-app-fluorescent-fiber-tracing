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

def contour(im, filename, vmin, vmax, cmap):
    #Determine where to draw contours (levels) – 1%, 5%, 10%, 20%, 40%, 60%,
    #   80% (and 100% when the density's maximum value is equal to vmax: a
    #   work around matplotlib.pyplot.contourf()'s 'color fill' and 'levels'
    #   mapping). 
    #https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html#matplotlib.pyplot.contourf

    #Save a blank image if density is zero.
    shp = np.array((im.shape[1],im.shape[0]))
    dpi = 72
    if np.max(im)==0:
        plt.figure(figsize=tuple(shp/dpi))
        plt.axis('off')
        plt.savefig(filename)
        plt.clf()
        return
    
    #Else, plot and save contours    
    densityRange = np.max(im)-np.min(im)
    '''lvls = np.min(im) + (densityRange*np.array((0.01,0.05,0.1,0.2,0.4,
                                                0.6,0.8,0.9,0.95,1.0)))'''
    lvls = np.min(im) + (densityRange*np.array((0.01, 0.15,1.0)))                                            
    if np.max(im)==vmax: lvls=lvls[:-1] #remove highest level 

    x = np.linspace(0, im.shape[1], im.shape[1])
    y = np.linspace(0, im.shape[0], im.shape[0])
    X, Y = np.meshgrid(x,y)
    
    plt.figure(figsize=tuple(shp/dpi))
    plt.contourf(X, -Y, im, levels=lvls, cmap=cmap, 
                 origin='image', vmax=vmax, vmin=vmin, extend='both')
    plt.axis('off')
    plt.savefig(filename)
    plt.clf() 

def fiberDensityMap(im, sigma):
    if np.max(im)==0: return im
    #print(im.dtype)
    return gaussian_filter(im,sigma)

def simplify_svg(fileName):
    file = open(fileName,'r')
    fileContent = file.readlines()
    file.close()
    text = ""
    for i,line in enumerate(fileContent):
        text+=line 
    fileContent = text.split('<')
    text = ""
    for i,line in enumerate(fileContent): 
        if 'xml version' in line or 'svg xmlns' in line:
            text += "<"+line
        if "PathCollection_1" in line:
            for j,line in enumerate(fileContent[i:]):
                if "PathCollection_2" in line: break
                if 'clip-path' in line:
                    text += '<' + line[:-36] + '/>\n' #"<path"+line[34:]
            #break
        if '/svg>' in line:
            text += "<"+line
        if 'rect ' in line:
            text += "<"+line[:-5]+' fill="none" stroke="black" stroke-width="1px"/>\n'
        else: continue
    file = open(fileName[:-4]+'.svg', 'w')
    file.write(text)
    file.close()

def run(img_path, channel, raw_output, thrs_output, contour_output, skeleton_output, unet_agrp, thr):
    t = time()

    conf = Dict(yaml.safe_load(open('segmentation/utils/unet_predict.yaml')))
    model_path = 'segmentation/model_bestVal.pt' 

    print(f'Loading image')  
    filename = img_path.split(".")[0]
    img = mpimg.imread(img_path)
    _img = verifyChannel(img, channel)
    _img = verifyDims(_img, conf["window_size"])
    _img = verifyRange(_img)
    y = _img+0.0

    if unet_agrp=='1':
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
        
        y = y[0:img.shape[0],0:img.shape[1]]
    
    if raw_output=='1':
        print(f'Saving raw prediction')
        plt.imsave(filename+'_raw.png', y, cmap='gray', vmin=0, vmax=1)

    y = y > thr
    print(y.shape, np.max(y))
    y = remove_small_objects(y, 9,connectivity=8).astype(int)

    if thrs_output=='1':
        print(f'Saving threshold prediction')
        plt.imsave(filename+'_thrs.png', y.astype('uint8')*255, cmap='gray', vmin=0, vmax=1)

    if contour_output=='1':
        print(f'Creating Contour from Model Prediction')
        matplotlib.pyplot.switch_backend('Agg') 
        sigma = 0
        im = 1-y
        print("\nObtaining Contour ")
        density = fiberDensityMap(im,sigma)
        #print('\tIm shape', density.shape)
        #print('\tRange:', np.min(density), np.max(density))
        outFileName = f'{filename}_contour.svg'
        #print("\nGenrating Isopleth Contours")
        contour(im, outFileName, 
                vmin = 0, vmax = np.max(im), cmap = 'gray') #newColorMap('gray', nColors=1000, reverse=True, opacity=False))
        print('Saving Contour')
        print("\nSimplifying SVG file %s"%outFileName)
        simplify_svg(outFileName)

    if skeleton_output=='1':
        print(f'Skeletonizing Model Output')
        skeleton = skeletonize(y).astype('uint8')*255
        imageio.imwrite(filename+'_skl.png', skeleton) #saving skeleton for vectorization c program
        print(f'Vectorizing Skeleton')
        polylines = check_output(["segmentation/a.out", filename+'_skl.png', "to", "a"]) #running c program as a subprocess
        os.remove(filename+'_skl.png')
        p = polylines.decode("utf-8") 
        p = p.split('\n')[:-1]
        writeSVG(filename+'_skl.svg',p,y.shape[0], y.shape[1])
    
    t = time()-t
    print(f'Finished segmenting and tracing {filename} w dims {y.shape} in {t} sec\n') 

    