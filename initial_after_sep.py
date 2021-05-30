#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:10:19 2020

@author: stanton
"""

"""
Created on Mon Apr 20 21:11:49 2020

@author: yjiang
"""
import os
import glob
import xlrd
import re
import string
import math
from random import randint
import csv
import pydicom
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim  
import matplotlib 
#matplotlib.use('Agg') #远程时非图形界面可避免出错
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
from collections.abc import Iterable
from torchsummaryX import summary
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology
from skimage import segmentation
from skimage import color
np.set_printoptions(threshold=np.inf)

citybook = xlrd.open_workbook('/mnt/4E0AC9410AC926B5/Linux/COVID/German/districtgrid.xlsx')
citycasesbook = xlrd.open_workbook('/mnt/4E0AC9410AC926B5/Linux/COVID/German/German_Data_1125.xlsx')

citymap = citybook.sheet_by_index(0)
citycases = citycasesbook.sheet_by_index(0)

cityidlist=[citymap.col_values(3), citymap.col_values(4)]
citygrid=np.array(cityidlist)
citynum=citygrid.shape[1]

citycaseslist=[citycases.col_values(3), citycases.col_values(6), citycases.col_values(10)]
citycasesgrid=np.array(citycaseslist)
totaldays=int(((citycasesgrid.shape)[1]-1)/citynum)
# print(totaldays)
# print(citycasesgrid[:, 83])
# print(citycasesgrid[:, 83+totaldays])

cityset=[]

studynumdays=110
germanboard=np.zeros([25, 25])-100
dailygridcases=np.zeros([79, 25, 25])

for j in range(citynum):
    cityset.append(math.floor(float(citycasesgrid[2, 83+j*totaldays])))

    
for idx in range(citynum):
    cityid=int(re.findall(r"\d+\.?\d*", citygrid[0, idx])[0]) #cityid as a number
    #print(idx+1, cityid)
    idxcityid=cityset.index(cityid)
    dailycitycases=citycasesgrid[0, 250+idxcityid*totaldays:329+idxcityid*totaldays]#still strings
    
    grididlist=((citygrid[1, idx]).replace("/", ".")).split(";")

    for loc in grididlist:
        rowidstr, colidstr, ratiostr=re.findall(r"\d+\.?\d*", loc)
        rowid=int(rowidstr)
        colid=int(colidstr)
        rationum=float(ratiostr) #row, col, ratio as numbers
        germanboard[rowid, colid]=1
        
        if(rationum>1.0):
            ratio=math.floor(rationum)/(10*(rationum-math.floor(rationum)))
        else:
            ratio=rationum
        for jd in range(79):
            dailygridcases[jd, rowid, colid]+=ratio*float(dailycitycases[jd])
            
        #print(rowid, colid, ratio)

# for it in range(30):
#     print(np.sum(dailygridcases[it, :, :]))

# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numbers')
# plt.show()
    
# fig = plt.figure()
# #定义画布为1*1个划分，并在第1个位置上进行作图
# # labels=[]
# # for i in range(9): 
# #     labels.append(3*i)
# ax = fig.add_subplot(111)
# #定义横纵坐标的刻度
# # ax.set_yticks(range(25))
# # ax.set_yticklabels(labels)
# # ax.set_xticks(range(25))
# # ax.set_xticklabels(labels)
# #作图并选择热图的颜色填充风格，这里选择hot
# im = ax.imshow(np.log(1+dailygridcases[10, :, :]), cmap=plt.cm.get_cmap('hot_r'))
# #print(np.sum(dailygridcases[10, :, :]))
# #增加右侧的颜色刻度条
# plt.colorbar(im)

# #增加标题
# plt.title("Real Profile@20th day")
# #show
# plt.show()



   
np.random.seed(0)
#studynumdays=30#to be comment finally
ep=0.09 #0.09/2.0 #0.09/8 #0.09
l1=0.21 #0.21/2.0 #0.21/8 #0.21 #0.17
p13=0.17 #0.17/2.0 #0.17/8 #0.17 #0.13
pcd=0.0628 #0.628/2.0 #0.628/8 #0.628

dailymodelexpose=np.zeros([studynumdays, 25, 25])
dailymodelcases=np.zeros([studynumdays, 25, 25])
summodelcases=np.zeros([studynumdays, 25, 25])

pl=np.ones([25, 25])

pr=np.ones([25, 25])

pu=np.ones([25, 25])

pd=np.ones([25, 25])


for ir in range(24):
    for ic in range(25):
        if (germanboard[ir+1, ic]>=0 and germanboard[ir, ic]<0):
            pu[ir+1, ic]=0
        if (germanboard[ir, ic]>=0 and germanboard[ir+1, ic]<0):
            pd[ir, ic]=0
            
for ir in range(25):
    for ic in range(24):
        if (germanboard[ir, ic+1]>=0 and germanboard[ir, ic]<0):
            pl[ir, ic+1]=0
        if (germanboard[ir, ic]>=0 and germanboard[ir, ic+1]<0):
            pr[ir, ic]=0

pl[:, 0]=0
pr[:, 24]=0
pu[0, :]=0
pd[24, :]=0
            
for ir in range(25):
    for ic in range(25):
        totalpos=pl[ir, ic]+pr[ir, ic]+pu[ir, ic]+pd[ir, ic]
        pl[ir, ic]=pl[ir, ic]/totalpos
        pr[ir, ic]=pr[ir, ic]/totalpos
        pu[ir, ic]=pu[ir, ic]/totalpos
        pd[ir, ic]=pd[ir, ic]/totalpos
        if germanboard[ir, ic]<0:
            pl[ir, ic]=0
            pr[ir, ic]=0
            pu[ir, ic]=0
            pd[ir, ic]=0
        
#initialization is needed here

dailymodelexpose[0, :, :]=23*0.025*dailygridcases[0, :, :] # 23 from 9-8 to 11-25  
# dailymodelexpose[0, 11, 3]=dailymodelexpose[0, 11, 3]-50
# dailymodelexpose[0, 12, 3]=dailymodelexpose[0, 12, 3]-45
# dailymodelexpose[0, 2, 9]=dailymodelexpose[0, 2, 9]+2
# dailymodelexpose[0, 2, 10]=dailymodelexpose[0, 2, 10]+2
# dailymodelexpose[0, 2, 12]=dailymodelexpose[0, 2, 12]+2
# dailymodelexpose[0, 5, 12]=dailymodelexpose[0, 5, 12]+20
# dailymodelexpose[0, 6, 14]=dailymodelexpose[0, 6, 14]+10
# dailymodelexpose[0, 8, 20]=dailymodelexpose[0, 8, 20]+10
# dailymodelexpose[0, 9, 16]=dailymodelexpose[0, 9, 16]+5
# dailymodelexpose[0, 10, 17]=dailymodelexpose[0, 10, 17]+5
# dailymodelexpose[0, 11, 9]=dailymodelexpose[0, 11, 9]+10
# dailymodelexpose[0, 11, 11]=dailymodelexpose[0, 11, 11]+10
# dailymodelexpose[0, 11, 18]=dailymodelexpose[0, 11, 18]+10
# dailymodelexpose[0, 12, 14]=dailymodelexpose[0, 12, 14]+5
# dailymodelexpose[0, 12, 18]=dailymodelexpose[0, 12, 18]+5
# dailymodelexpose[0, 13, 13]=dailymodelexpose[0, 13, 13]+5
# dailymodelexpose[0, 13, 18]=dailymodelexpose[0, 13, 18]+5
# dailymodelexpose[0, 14, 9]=dailymodelexpose[0, 14, 9]+10
# dailymodelexpose[0, 14, 16]=dailymodelexpose[0, 14, 16]+5
# dailymodelexpose[0, 18, 10]=dailymodelexpose[0, 18, 10]+20
# dailymodelexpose[0, 20, 7]=dailymodelexpose[0, 20, 7]+5

dailymodelcases[0, :, :]=1.01*dailygridcases[78+0, :, :] # 78 from 9-8 to 11-25 
summodelcases[0, :, :]=1.01*dailygridcases[78+0, :, :] # 78 from 9-8 to 11-25
print(dailygridcases[78+0, 14, 9]) 

for time in range(studynumdays-1):
    
    dailymodelexpose[time+1, :, :]=np.maximum(0, dailymodelexpose[time, :, :]+(l1-p13)*dailymodelexpose[time, :, :])
    #dailymodelcases[time+1, :, :]=np.maximum(0, dailymodelcases[time, :, :]+p13*dailymodelexpose[time, :, :]-pcd*dailymodelcases[time, :, :])
    summodelcases[time+1, :, :]=summodelcases[time, :, :]+p13*dailymodelexpose[time, :, :]
            
    for ir in range(25):
        for ic in range(25):
            
            compass=np.random.choice([0, 1, 2, 3, 4], p = [pl[ir, ic], pr[ir, ic], pu[ir, ic], pd[ir, ic], 1-pl[ir, ic]-pr[ir, ic]-pu[ir, ic]-pd[ir, ic]])

            if compass==0:
                #dailymodelcases[time+1, ir, ic-1]=dailymodelcases[time+1, ir, ic-1]+ep*dailymodelcases[time, ir, ic]
                dailymodelexpose[time+1, ir, ic-1]=dailymodelexpose[time+1, ir, ic-1]+ep*dailymodelexpose[time, ir, ic]
            elif compass==1:
                #dailymodelcases[time+1, ir, ic+1]=dailymodelcases[time+1, ir, ic+1]+ep*dailymodelcases[time, ir, ic]
                dailymodelexpose[time+1, ir, ic+1]=dailymodelexpose[time+1, ir, ic+1]+ep*dailymodelexpose[time, ir, ic]
            elif compass==2:
                #dailymodelcases[time+1, ir-1, ic]=dailymodelcases[time+1, ir-1, ic]+ep*dailymodelcases[time, ir, ic]
                dailymodelexpose[time+1, ir-1, ic]=dailymodelexpose[time+1, ir-1, ic]+ep*dailymodelexpose[time, ir, ic]
            elif compass==3:
                #dailymodelcases[time+1, ir+1, ic]=dailymodelcases[time+1, ir+1, ic]+ep*dailymodelcases[time, ir, ic]
                dailymodelexpose[time+1, ir+1, ic]=dailymodelexpose[time+1, ir+1, ic]+ep*dailymodelexpose[time, ir, ic]
                
            #dailymodelcases[time+1, ir, ic]=dailymodelcases[time+1, ir, ic]-ep*dailymodelcases[time, ir, ic]
            dailymodelexpose[time+1, ir, ic]=dailymodelexpose[time+1, ir, ic]-ep*dailymodelexpose[time, ir, ic]
            

# fig = plt.figure()
# #定义画布为1*1个划分，并在第1个位置上进行作图

# ax = fig.add_subplot(111)
# #定义横纵坐标的刻度
# # ax.set_yticks(range(25))
# # ax.set_yticklabels(labels)
# # ax.set_xticks(range(25))
# # ax.set_xticklabels(labels)
# #作图并选择热图的颜色填充风格，这里选择hot
# im = ax.imshow(np.log(1+np.ceil(summodelcases[10, :, :])), cmap=plt.cm.get_cmap('hot_r'))
# #增加右侧的颜色刻度条
# plt.colorbar(im)
# #增加标题
# plt.title("Simulated Profile@20th day")
# #show
# plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 1, 1)
plt.imshow(np.log10(1+np.ceil(dailygridcases[29, :, :])), cmap='hot_r')
plt.colorbar()
plt.title("Real Profile@30th day")
plt.clim(0, 4)

plt.figure(figsize=(10, 4))

plt.subplot(1, 1, 1)
plt.imshow(np.log10(1+np.ceil(summodelcases[29, :, :])), cmap='hot_r')
plt.colorbar()
plt.title("Simulated Profile@30th day (Unlocked)")
plt.clim(0, 4)

trainfile = open('COVIDMAPS_kai_unlock_11_testv.dat','w')
for train_days in range(studynumdays):
    for ir in range(25):
        for ic in range(25):
            trainfile.write(str(np.ceil(summodelcases[0+train_days, ir, ic]))+' ')
    trainfile.write(str(ep)+' '+str(l1)+' '+str(p13)+' '+str(pcd)+' '+str(np.ceil(np.sum(dailymodelexpose[0+train_days, :, :])))+' '+str(0+train_days)+'\n')
        
trainfile.close()  
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(np.log10(1+np.ceil(dailygridcases[10, :, :])), cmap='hot_r')
# plt.colorbar()
# plt.title("Real Profile@10th day")
# plt.clim(0, 3)

# plt.subplot(1, 2, 2)
# plt.imshow(np.log10(1+np.ceil(summodelcases[10, :, :])), cmap='hot_r')
# plt.colorbar()
# plt.title("Simulated Profile@10th day")
# plt.clim(0, 3)

# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(np.log10(1+np.ceil(dailygridcases[20, :, :])), cmap='hot_r')
# plt.colorbar()
# plt.title("Real Profile@20th day")
# plt.clim(0, 3)

# plt.subplot(1, 2, 2)
# plt.imshow(np.log10(1+np.ceil(summodelcases[20, :, :])), cmap='hot_r')
# plt.colorbar()
# plt.title("Simulated Profile@20th day")
# plt.clim(0, 3)

# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(np.log10(1+np.ceil(dailygridcases[29, :, :])), cmap='hot_r')
# plt.colorbar()
# plt.title("Real Profile@30th day")
# plt.clim(0, 3)

# plt.subplot(1, 2, 2)
# plt.imshow(np.log10(1+np.ceil(summodelcases[29, :, :])), cmap='hot_r')
# plt.colorbar()
# plt.title("Simulated Profile@30th day")
# plt.clim(0, 3)
# for it in range(30):
#     print(np.sum(summodelcases[it, :, :]))

#print(np.sum(summodelcases[0, :, :]))

labels=[]
for i in range(30): 
    labels.append(i+1)

figr=[]
figs=[]

for it in range(30):
    figs.append(np.sum(summodelcases[it, :, :]))
    figr.append(np.sum(dailygridcases[it, :, :]))
    
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(labels, figr, color = 'r') 
plt.scatter(labels, figs, color = 'b')
plt.show()

# dailymodelexpose[0, 6, 9]=10
# dailymodelexpose[0, 11, 3]=110
# dailymodelexpose[0, 11, 20]=10
# dailymodelexpose[0, 12, 3]=100
# dailymodelexpose[0, 12, 5]=20
# dailymodelexpose[0, 15, 14]=10
# dailymodelexpose[0, 16, 5]=10
# dailymodelexpose[0, 17, 10]=10
# dailymodelexpose[0, 18, 9]=5
# dailymodelexpose[0, 18, 11]=5
# dailymodelexpose[0, 18, 12]=5
# dailymodelexpose[0, 18, 10]=25
# dailymodelexpose[0, 19, 14]=10
# dailymodelexpose[0, 19, 8]=5
# dailymodelexpose[0, 19, 9]=5
# dailymodelexpose[0, 19, 16]=40
# dailymodelexpose[0, 19, 15]=30
# dailymodelexpose[0, 20, 7]=30
# dailymodelexpose[0, 20, 18]=40
# dailymodelexpose[0, 20, 14]=10
# dailymodelexpose[0, 20, 15]=20
# dailymodelexpose[0, 20, 11]=5
# dailymodelexpose[0, 20, 12]=5
# dailymodelexpose[0, 20, 16]=10

# #Berlin
# dailymodelexpose[0, 8, 20]=77
# dailymodelexpose[0, 8, 21]=78
# #Hamburg
# dailymodelexpose[0, 5, 12]=81
# #Munchen
# dailymodelexpose[0, 19, 16]=60
# #Koln
# dailymodelexpose[0, 12, 5]=48
# #Frankfurt am main
# dailymodelexpose[0, 14, 9]=30
# #Stuttgart
# dailymodelexpose[0, 18, 10]=27
# #Dusseldorf
# dailymodelexpose[0, 11, 5]=27
# #Dortmund
# dailymodelexpose[0, 10, 6]=27
# #Essen
# dailymodelexpose[0, 11, 5]=dailymodelexpose[0, 11, 5]+27
# #Bremen
# dailymodelexpose[0, 6, 9]=21
# #Dresden
# dailymodelexpose[0, 11, 21]=21
# #Leipzig
# dailymodelexpose[0, 11, 18]=21
# #Hannover
# dailymodelexpose[0, 8, 11]=11
# dailymodelexpose[0, 8, 12]=11
# #Nurnberg
# dailymodelexpose[0, 16, 15]=22

# dailymodelexpose[0, 5, 12]=30
# dailymodelexpose[0, 6, 9]=20
# dailymodelexpose[0, 8, 11]=10
# dailymodelexpose[0, 8, 12]=10
# dailymodelexpose[0, 8, 20]=20
# dailymodelexpose[0, 8, 21]=20
# dailymodelexpose[0, 10, 6]=30
# dailymodelexpose[0, 11, 3]=30
# dailymodelexpose[0, 11, 5]=40
# dailymodelexpose[0, 11, 18]=10
# dailymodelexpose[0, 11, 20]=20
# dailymodelexpose[0, 11, 21]=10
# dailymodelexpose[0, 12, 3]=30
# dailymodelexpose[0, 12, 5]=40
# dailymodelexpose[0, 14, 9]=5
# dailymodelexpose[0, 15, 14]=5
# dailymodelexpose[0, 16, 5]=5
# dailymodelexpose[0, 16, 15]=5
# dailymodelexpose[0, 17, 10]=30
# dailymodelexpose[0, 18, 9]=20
# dailymodelexpose[0, 18, 10]=30
# dailymodelexpose[0, 18, 11]=20
# dailymodelexpose[0, 18, 12]=30
# dailymodelexpose[0, 19, 8]=10
# dailymodelexpose[0, 19, 9]=20
# dailymodelexpose[0, 19, 14]=10
# dailymodelexpose[0, 19, 15]=20
# dailymodelexpose[0, 19, 16]=30
# dailymodelexpose[0, 20, 7]=10
# dailymodelexpose[0, 20, 11]=10
# dailymodelexpose[0, 20, 12]=10
# dailymodelexpose[0, 20, 14]=10
# dailymodelexpose[0, 20, 15]=20
# dailymodelexpose[0, 20, 16]=20
# dailymodelexpose[0, 20, 18]=10

# iexpose=0

# while iexpose<40:
#     ix=randint(5, 20)
#     iy=randint(5, 20)
#     if dailymodelexpose[0, ix, iy]>0:
#         iexpose-=1
#     else:
#         dailymodelexpose[0, ix, iy]=0
#     iexpose+=1    