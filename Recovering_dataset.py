#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:17:45 2022

@author: aghm
"""

import numpy as np
import pandas as pd


label=pd.read_csv('label_server.csv') 
label=label.loc[:, ~label.columns.str.contains('^Unnamed')]
print(label.columns)
fbp_array=[0.3,0.5,0.7,0.9]

aux4=[]
for fbp in fbp_array:
    mat=[]
    mat2=[]
    mat3=[]
    # l=0
    out2=[]
    out2=pd.DataFrame(columns=[label.columns])
    # 1/0
    for att in label.columns:
        # print(att)
        ######
        aux2=[]
        aux2=pd.read_csv('results/_4D'+att+'_flopprob_'+str(fbp)+'_peratt_1000_probabilities_centroid.csv')
        aux2=aux2.loc[:, ~aux2.columns.str.contains('^Unnamed')]
        mat2.append(np.array(aux2.idxmax(axis=1)))
        
        aux3=[]
        aux3=pd.read_csv("results/perturb_"+att+'_flopprob_'+str(fbp)+'_peratt_1000'+'_'+'.csv')
        aux3=aux3.loc[:, ~aux3.columns.str.contains('^Unnamed')]
        out2[att] =aux3.values.tolist()
        # if l==1:
        #     aux4.append(aux3.values.tolist())
        # if l==0:
        #     aux4=aux3.values.tolist()
        #     l=1
        # print(aux4)
    # mat3.append(np.array(aux4))

        
    ########
    b=[]
    b=np.array(mat2)    
    b=np.transpose(np.array(mat2))
    out=[]
    out=pd.DataFrame(b,columns=[label.columns])
    out.to_csv('csv_eval/VAE_FULLSPACE_flopprob_'+str(fbp)+'_.csv')
    #####
    # b=[]
    # b=np.array(mat3)    
    # b=np.transpose(np.array(mat3))
    # out=[]
    # out=pd.DataFrame(mat3.T,columns=[label.columns])
    out2.to_csv('csv_eval/perturbed_dataset_'+str(fbp)+'_.csv')

###################################