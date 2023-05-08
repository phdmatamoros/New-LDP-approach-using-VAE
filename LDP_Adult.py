#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:10:31 2022

@author: aghm
"""
#lIBRARIES
from math import log
import math
from sklearn.linear_model import ARDRegression
from pure_ldp.frequency_oracles import *
from pure_ldp.heavy_hitters import *
from pure_ldp.core import generate_hash_funcs
from sklearn.preprocessing import PolynomialFeatures
from numpy import linalg as LA
import pandas as pd
import numpy as np
import random 
import secrets
import math
#from bloom_aghm import BloomFilter
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import matplotlib.ticker as mtick
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import init
warnings.filterwarnings('ignore')
from numpy import linalg as LA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
import seaborn as sb
import gc

import scipy as sp
from numpy import inf
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ARDRegression, LinearRegression, BayesianRidge

from aghmJan18Functions import *

from itertools import product  
from sklearn.utils.extmath import cartesian



df = pd.read_csv('Votes_label_server.csv')
df=df.loc[:, ~df.columns.str.contains('^Unnamed')]

dk = pd.read_csv('Votes_hash_server.csv')
dk=dk.loc[:, ~dk.columns.str.contains('^Unnamed')]

savedf=[]
savedf=df
for col in savedf.columns:
    savedf[col] = savedf[col].astype('category')
cat_columns=savedf.select_dtypes(['category']).columns
savedf[cat_columns] = savedf[cat_columns].apply(lambda x: x.cat.codes)
savedf.to_csv('categorical.csv',index=False)


#########################
#########################
#########################



fabp=[0.1,0.4,0.7,0.9]
valm=128

combinations=100
sampling=1################################.1
regression="Lasso"

sampling=sampling
df = pd.read_csv('Votes_label_server.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

dfcat = pd.read_csv('categorical.csv')
# dfcat = dfcat.loc[:, ~df.columns.str.contains('^Unnamed')]

less128=[]
tamless128=[]
BloomFilter={}
BloomSize={}

shi=0
acushi=0
for col in df.columns:
    
    if len(str2vec(df[col][0]))<=valm:#128
        less128.append(col)
        tamless128.append(len(str2vec(df[col][0])))
        BloomSize['{}'.format(col)]=len(df[col].unique())
        BloomFilter['{}'.format(col)]=len(str2vec(df[col][0]))
        acushi=acushi+len(df[col].unique())
        shi=shi+1
ave_car=acushi/shi
#################    
df = pd.read_csv('categorical.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df[less128]
poc=df.corr()
cua1=[]
cua1=pd.DataFrame(poc)
cua1.columns =[less128]
cua1.index = [less128]
sb.heatmap(cua1, cmap="YlGnBu", annot=False, vmin=-1,vmax=1)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("Original_Correlations.png")
plt.close()

df = []
df = pd.read_csv('Votes_label_server.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df[less128]

mais=len(df)
value_real={}
value_cat={}
codified={}
for col in df.columns:
    value_real['{}'.format(col)]=list(df[col].unique())
    value_cat['{}'.format(col)]=list(dfcat[col].unique())
    value_cat['{}'.format(col)]=list(dfcat[col].unique())
    codified['{}'.format(col)]=list(np.array(dfcat[col].unique())+1)
for fbp in fabp:
    kways=[1,2,3,4,5]
    #######
    kway1_A=[]
    kway1_B=[]
    kway1_C=[]
    kway1_D=[]
    kway1_E=[]
    kway1_F=[]
    kway1_G=[]
    kway1_H=[]
    kway1_I=[]
    
    kway2_A=[]
    kway2_B=[]
    kway2_C=[]
    kway2_D=[]
    kway2_E=[]
    kway2_F=[]
    kway2_G=[]
    kway2_H=[]
    kway2_I=[]
    
    kway3_A=[]
    kway3_B=[]
    kway3_C=[]
    kway3_D=[]
    kway3_E=[]
    kway3_F=[]
    kway3_G=[]
    kway3_H=[]
    kway3_I=[]
    
    kway4_A=[]
    kway4_B=[]
    kway4_C=[]
    kway4_D=[]
    kway4_E=[]
    kway4_F=[]
    kway4_G=[]
    kway4_H=[]
    kway4_I=[]
    
    kway5_A=[]
    kway5_B=[]
    kway5_C=[]
    kway5_D=[]
    kway5_E=[]
    kway5_F=[]
    kway5_G=[]
    kway5_H=[]
    kway5_I=[]
    
    
    # 
    for times in range(0,combinations):
        lassoF=[]
        lassomF=[]
        locopF=[]
        locopmF=[]
        df_cor = df.head(int(len(df)*sampling)) #sample(frac=sampling,random_state=times)
        ############Perturbing Original dataset 
        ####################################################################################
        bit = pd.read_csv('csv_eval/perturbed_dataset_'+str(fbp)+'_.csv')
        bit =  bit.loc[:, ~bit.columns.str.contains('^Unnamed')]

        attributes=less128
        ####################################################################################
        random.seed(times)
        attributes_full=random.sample(less128,len(kways))
        PDM=0
        noPDM=0
        distance_matrix=[]
        print(times,attributes_full)
    
        for kway in kways:

            gc.collect()
            attributes=attributes_full[0:kway]

            #if kway==kways[0]:
            bit2 = bit[attributes]
            bitM = bit[attributes]
            
            bit2 = bit2.head(int(len(df)*sampling)) #.sample(frac=sampling,random_state=times)
            bitM = bitM.head(int(len(df)*sampling)) #.sample(frac=sampling,random_state=times)

            #####Binary
            bit_perturb = df2nperturb(bit2,fbp)
            # #####Hexal
            # bit_perturb = dftohexal(bit2,fbp)
            # bit_perturb = df2nperturbhexal(bit_perturb,fbp)
            # bit_perturb = df2hexaltobin(bit_perturb)
            # bit_perturb = hexalfinal(bit_perturb)
            # dondegu='/resultshexal/'
            # #####

            
            bit_si=df2size(bit2)
            bit_size=np.cumsum(df2size(bit2))
            dict_perturb={}
            # print(bit_size)
            for att,indexa in zip(attributes,range(len(attributes))):
            #for att,indexa in zip(attributes_full,range(len(attributes_full))):
                # print(att,indexa)
                if 0==indexa:
                    a=bit_size[indexa]-1
                    b=bit_size[indexa+1]-1
                if 0!=indexa:
                    a=bit_size[indexa]-1
                    b=bit_size[indexa+1]-1
                dict_perturb['{}'.format(att)]=bit_perturb[:,a:b]
            
            lassoF=[]
            lassomF=[]
            locopF=[]
            locopmF=[]
            #attributes=attributes_full[0:kway]
            d2 = dfcat[attributes]
            mana = []
            mana = d2.head(int(len(df)*sampling)) #.sample(frac=sampling,random_state=times)
            ################################################REAL
            # print("REAL-computing Joint distributions")
            fq_real = mana.value_counts()
            p_real = fq_real/fq_real.sum()
            p_real=p_real.to_frame()    
            p_real.columns = ["Real"]
          
            # ################################################LOPUB-LASSO
            p_lasso=[]
            p_lasso,nu=LASSO_original(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,valm,valor,hashfun)
            # ################################################LASSO-V1
            p_br=[]
            p_br,nu=Br_original(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,valm,valor,hashfun)
            

            L_locop_A=[]
            locopa=[]

            L_locop_A=[]
            locopb=[]
            L_locop_A=[]
            locopc=[]
            
            ###################################################################
            ###################################################################
            ###################################################################
            ###################################################################
            ###################################################################
            ###################################################################
            ###########LOCOP
            dfmul=[]
            dfmulQ=[]
            ant=[]
            lind=[]
            ii=[]
            tuples=[]
            index=[]
            for poi in attributes:
                dfmul.append(value_cat[poi])
        
            if kway==1:
                dfmulQ = pd.DataFrame(dfmul[0])
                dfmulQ.columns = [attributes[0]]
                dfmulQ =(list(product(dfmulQ[attributes[0]])))
                lista=[]
                for v in dfmulQ:
                    if sum(np.isnan(v)) <= 0:
                        lista.append(v)
                ii=np.array(lista)
                tuples=[ii[:,0]]
                tuples = list(zip(*tuples))
                index = pd.MultiIndex.from_tuples(tuples, names=attributes)
                
                ant=pd.DataFrame(lista) 
                ant.columns = [attributes[0]]
                ant = ant.astype('int')
                matrix=np.eye(1, dtype=int)
                
    
                
            if kway==2: 
                dfmulQ = pd.DataFrame((dfmul[0],dfmul[1])).T
                dfmulQ.columns = [attributes[0],attributes[1]]
                dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]])))
                lista=[]
                for v in dfmulQ:
                    if sum(np.isnan(v)) <= 0:
                        lista.append(np.array(v))
                ii=np.array(lista)
                tuples=[ii[:,0],ii[:,1]]
                tuples = list(zip(*tuples))
                index = pd.MultiIndex.from_tuples(tuples, names=attributes)
                
                ant=pd.DataFrame(lista) 
                ant.columns = [attributes[0],attributes[1]]
                ant = ant.astype('int')
    
                
            if kway==3:
                dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2])).T
                dfmulQ.columns = [attributes[0],attributes[1],attributes[2]]
                dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]])))
                lista=[]
                for v in dfmulQ:
                    if sum(np.isnan(v)) <= 0:
                        lista.append(v)
                ii=np.array(lista)
                tuples=[ii[:,0],ii[:,1],ii[:,2]]
                tuples = list(zip(*tuples))
                index = pd.MultiIndex.from_tuples(tuples, names=attributes)
                ant=pd.DataFrame(lista) 
                ant.columns = [attributes[0],attributes[1],attributes[2]]
                ant = ant.astype('int')
    
            if kway==4:
                dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3])).T
                dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
                dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]])))
                lista=[]
                for v in dfmulQ:
                    if sum(np.isnan(v)) <= 0:
                        lista.append(v)
                ii=np.array(lista)
                tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3]]
                tuples = list(zip(*tuples))
                index = pd.MultiIndex.from_tuples(tuples, names=attributes)
                ant=pd.DataFrame(lista) 
                ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3]]
                ant = ant.astype('int')
                
            if kway==5:
                dfmulQ = pd.DataFrame((dfmul[0],dfmul[1],dfmul[2],dfmul[3],dfmul[4])).T
                dfmulQ.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
                dfmulQ =(list(product(dfmulQ[attributes[0]], dfmulQ[attributes[1]],dfmulQ[attributes[2]],dfmulQ[attributes[3]],dfmulQ[attributes[4]])))
                lista=[]
                for v in dfmulQ:
                    if sum(np.isnan(v)) <= 0:
                        lista.append(v)
                ii=np.array(lista)
                tuples=[ii[:,0],ii[:,1],ii[:,2],ii[:,3],ii[:,4]]
                tuples = list(zip(*tuples))
                index = pd.MultiIndex.from_tuples(tuples, names=attributes)
                ant=pd.DataFrame(lista) 
                ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
                ant = ant.astype('int')
            final_index=index
    
            ############
            aux_value={}
            miu_value={}
            marginal_dist={}
            marginal_index={}
            marginal_index_out={}
            numeric={}
            list_att=list(d2.columns)
            for indexa in range(len(d2.columns)):
                att=list_att[indexa]
                att2=[list_att[indexa]]
                ##############################################################
                dk=[]
                dk = df[att]
                nu,p_est=LASSO_original(d2, att2,dict_perturb,1,bit,fbp,value_cat,df,valm,valor,hashfun)
                ###########
                bit4=dict_perturb[att]
                marginal_dist['{}'.format(att)]=p_est
                marginal_index_out['{}'.format(att)]=value_cat[att]
                vector=np.array(codified[att])              
                marginal_index['{}'.format(att)]=dk.value_counts().index
                miu_value['{}'.format(att)]=sum(marginal_dist[att]*vector) #calculating miu  Equation4 
                aux_value['{}'.format(att)]=sum(marginal_dist[att]*vector*len(bit4)) #calculating miu  Equation
            ############
            join_dist={}
            join_index={}
            join_index_full={}
            list_att=list(d2.columns)
            dist1=range(0,len(list_att)-1)
            # print("LOCOP-computing Joint distributions")
            for i in dist1:
                for j in range((i+1),len(list_att)):
                    attributes=[list_att[i],list_att[j]]
                    kw=len(attributes)
                    att2=list_att[i]+'&'+list_att[j]
                    ###############################LOCOP-joint Distribution
                    nu,p_t=LASSO_original(d2, attributes,dict_perturb,kw,bit,fbp,value_cat,df,valm,valor,hashfun)
                    df2=d2[attributes]
                    join_index['{}'.format(att2)]=list(df2.value_counts().index)
                    join_index_full['{}'.format(att2)]=list(d2.value_counts().index)
                    join_dist['{}'.format(att2)]=p_t    
            #########
            ###Computing pearson correlation coefficient matrix
            list_att=list(d2.columns)
            matrix=np.ones([len(list_att),len(list_att)])
            ori_cor=np.ones([len(list_att),len(list_att)])
            flag=0
            for k in range(len(list_att)-1):
                for v in range(k+1,len(list_att)):
                    #########Equation5
                    att1=list_att[k]
                    att2=list_att[v]
                    att=att1+'&'+att2
                    labelk=marginal_index[att1]
                    labelv=marginal_index[att2]
                    df2=d2[[att1,att2]]
                    fq_ori = df2.value_counts()
                    auxi=[]
                    auxi=fq_ori.index.to_numpy()
                    be=[]
                    for ele in auxi:
                        be.append(np.array(ele))
                    be=np.array(be)
                    p_aux=np.array(join_dist[att])
                    acu=0
                    for koin in range(len(marginal_index[att1])):
                        ko=koin+1
                        for voin in range(len(marginal_index[att2])):
                            vo=voin+1
                            if 1==sum(np.prod(be==[koin,voin],1)):
                                index=list(np.prod(be==[koin,voin],1)).index(1)
                                acu=acu+(p_aux[index]*ko*vo*len(bit4))
                            if 0==sum(np.prod(be==[koin,voin],1)):
                                acu=acu+(0*ko*vo)
                    ########Acu
                    #########Equation6
                    vector1=np.array(codified[att1])
                    vector2=np.array(codified[att2])
                    sumk=sum(marginal_dist[att1]*pow(vector1,2)*len(bit4))#Equation 6
                    sumv=sum(marginal_dist[att2]*pow(vector2,2)*len(bit4))#Equation 6
                    m1=miu_value[att1]
                    m2=miu_value[att2]
                    ###########################################################
                    A=acu
                    B=-miu_value[att1]*aux_value[att2]
                    C=-miu_value[att2]*aux_value[att1]
                    D=(len(bit4)*m1*m2)
                    up=A-D
                    down=pow( (sumk-(len(bit4)*pow(m1,2))) ,1/2) * pow( (sumv-(len(bit4)*pow(m2,2))) ,1/2)
                    ###########################################################
                    matrix[k,v]=up/down
                    matrix[v,k]=up/down
                    #############Real dataset
                    df_cor["uno"]=df_cor[att1].astype('category').cat.codes
                    df_cor["dos"]=df_cor[att2].astype('category').cat.codes
                    ori_cor[k,v]=df_cor['uno'].corr(df_cor['dos'])
                    ori_cor[v,k]=df_cor['uno'].corr(df_cor['dos'])
                    df_cor=df_cor.drop(['uno', 'dos'], axis=1)
            matrix[matrix ==  -inf] = -1
            matrix[matrix ==  inf] = 1
            where_are_NaNs = isnan(matrix)
            matrix[where_are_NaNs] = 0
            if np.all(np.linalg.eigvals(matrix) > 0):
                r=matrix
            else:
                from numpy import linalg as lg
                Eigenvalues, Eigenvectors = lg.eig(matrix)
                Lambda = np.diag(Eigenvalues)
                #print(Eigenvectors@Lambda@lg.inv(Eigenvectors))#Recover
                r_one=Eigenvectors@abs(Lambda)@lg.inv(Eigenvectors)
                v1=abs(r_one)*0
                for i in  range(len(matrix)):
                    v1[i,i]=pow(r_one[i,i],-.5)
                r=v1@r_one@v1
            alpha=ori_cor@r
            mat_corr=1-((np.trace(alpha)/(LA.norm(ori_cor, 'fro')*LA.norm(r, 'fro'))))
            ####Algoirthm 2
            cdf_A={}
            cdf_B={}
            cdf_C={}
            mean_d=[]
            for i in list(d2.columns):
                mi=np.cumsum(marginal_dist[i])
                cdf_B['{}'.format(i)]=np.hstack((np.cumsum(marginal_dist[i])))
                cdf_C['{}'.format(i)]=np.hstack(([0], np.cumsum(marginal_dist[i])))
                cdf_A['{}'.format(i)]=np.hstack(([0], mi[0:-1]))
                mean_d.append(np.mean((marginal_dist[i])))
            mean_ = np.ones(len(list(d2.columns)))
            percentage=.1
            syn_corrA=1
            syn_corrB=1
            try:
                copula_gaussian(n=int(len(df)*percentage), correlation=r,kwayr=kway)          
            except:
                r=np.eye(kway, dtype=int)
            a_val, a_cdf  = copula_gaussian(n=int(len(bit4)*percentage), correlation=r,kwayr=kway)
            t_val, t_cdf  = copula_gaussian_t(n=int(len(bit4)*percentage), correlation=r,kwayr=kway,freedoom=100)  
            #######
            synthetic=[]
            zzz=np.array(a_cdf).T
            out=[value_real]
            for y in zzz:
                out=[]
                for i,j in zip(list(d2.columns),y):
                    arr_B=cdf_A[i]
                    vector=marginal_index_out[i]
                    out.append((vector[sum(arr_B<=j)-1]))#-1
                synthetic.append(out)
            pd_syntheticL=[]
            pd_syntheticL=pd.DataFrame(synthetic)   
            pd_syntheticL.columns = [list(d2.columns)]  
            fq_syn = pd_syntheticL.value_counts()
            p_loco_L=fq_syn/fq_syn.sum()
            ######    
            arryb=[]
            for i in final_index:  
                if i in p_loco_L.index:
                    # print('aB')
                    arryb.append(p_loco_L[i])
                else:
                    # print('bB')
                    arryb.append(0)

            p_locop_L=pd.Series(list(arryb), index=final_index)
            p_locop_L=p_locop_L.to_frame()
            p_locop_L.columns = ["Locop"]
            ###################################################################
            ###################################################################
            ###################################################################
            ###################################################################
            ###################################################################
            ###################################################################
            df2=[]
            df2=p_real.join(p_lasso)
            df2=df2.join(p_br)
            df2=df2.join(p_locop_L)
            df3=df2.fillna(0)
            mi=[]
            for column in df3:
                mi.append(0.5*(df3[column] - df3['Real']).abs().sum())

            #mi.append(0)#Lasso
            #mi.append(0)#Br
            #mi.append(0)#Locop
            mi.append(0)#4
            mi.append(0)#5
            mi.append(0)#6
            mi.append(0)#7
            mi.append(0)#8
     

            if kway==1:
                kway1_A.append(mi[1])#Lasso
                kway1_B.append(mi[2])#Br
                kway1_C.append(mi[3])#Locop
                kway1_D.append(mi[4])#Lasso mod
                kway1_E.append(mi[5])#Br mod
                kway1_F.append(mi[6])#Br mod2
                kway1_G.append(mi[7])
                kway1_H.append(mi[8])
                kway1_I.append(0)
                
            if kway==2:
                kway2_A.append(mi[1])
                kway2_B.append(mi[2])
                kway2_C.append(mi[3])
                kway2_D.append(mi[4])
                kway2_E.append(mi[5])
                kway2_F.append(mi[6])
                kway2_G.append(mi[7])
                kway2_H.append(mi[8])
                kway2_I.append(0)
    
                
            if kway==3:
                kway3_A.append(mi[1])
                kway3_B.append(mi[2])
                kway3_C.append(mi[3])
                kway3_D.append(mi[4])
                kway3_E.append(mi[5])
                kway3_F.append(mi[6])
                kway3_G.append(mi[7])
                kway3_H.append(mi[8])
                kway3_I.append(0)
    
                
            if kway==4:
                kway4_A.append(mi[1])
                kway4_B.append(mi[2])
                kway4_C.append(mi[3])
                kway4_D.append(mi[4])
                kway4_E.append(mi[5])
                kway4_F.append(mi[6])
                kway4_G.append(mi[7])
                kway4_H.append(mi[8])
                kway4_I.append(0)
    
                
            if kway==5:
                kway5_A.append(mi[1])
                kway5_B.append(mi[2])
                kway5_C.append(mi[3])
                kway5_D.append(mi[4])
                kway5_E.append(mi[5])
                kway5_F.append(mi[6])
                kway5_G.append(mi[7])
                kway5_H.append(mi[8])
                kway5_I.append(0)
    
        if times==99:
            mkway1_A=np.mean(np.array(kway1_A))
            mkway1_B=np.mean(np.array(kway1_B))
            mkway1_C=np.mean(np.array(kway1_C))
            mkway1_D=np.mean(np.array(kway1_D))
            mkway1_E=np.mean(np.array(kway1_E))
            mkway1_F=np.mean(np.array(kway1_F))
            mkway1_G=np.mean(np.array(kway1_G))
            mkway1_H=np.mean(np.array(kway1_H))
            mkway1_I=np.mean(np.array(kway1_I))

            mkway2_A=np.mean(np.array(kway2_A))
            mkway2_B=np.mean(np.array(kway2_B))
            mkway2_C=np.mean(np.array(kway2_C))
            mkway2_D=np.mean(np.array(kway2_D))
            mkway2_E=np.mean(np.array(kway2_E))
            mkway2_F=np.mean(np.array(kway2_F))
            mkway2_G=np.mean(np.array(kway2_G))
            mkway2_H=np.mean(np.array(kway2_H))
            mkway2_I=np.mean(np.array(kway2_I))
            
            mkway3_A=np.mean(np.array(kway3_A))
            mkway3_B=np.mean(np.array(kway3_B))
            mkway3_C=np.mean(np.array(kway3_C))
            mkway3_D=np.mean(np.array(kway3_D))
            mkway3_E=np.mean(np.array(kway3_E))
            mkway3_F=np.mean(np.array(kway3_F))
            mkway3_G=np.mean(np.array(kway3_G))
            mkway3_H=np.mean(np.array(kway3_H))
            mkway3_I=np.mean(np.array(kway3_I))
            
            mkway4_A=np.mean(np.array(kway4_A))
            mkway4_B=np.mean(np.array(kway4_B))
            mkway4_C=np.mean(np.array(kway4_C))
            mkway4_D=np.mean(np.array(kway4_D))
            mkway4_E=np.mean(np.array(kway4_E))
            mkway4_F=np.mean(np.array(kway4_F))
            mkway4_G=np.mean(np.array(kway4_G))
            mkway4_H=np.mean(np.array(kway4_H))
            mkway4_I=np.mean(np.array(kway4_I))
            
            mkway5_A=np.mean(np.array(kway5_A))
            mkway5_B=np.mean(np.array(kway5_B))
            mkway5_C=np.mean(np.array(kway5_C))
            mkway5_D=np.mean(np.array(kway5_D))
            mkway5_E=np.mean(np.array(kway5_E))
            mkway5_F=np.mean(np.array(kway5_F))
            mkway5_G=np.mean(np.array(kway5_G))
            mkway5_H=np.mean(np.array(kway5_H))
            mkway5_I=np.mean(np.array(kway5_I))
            
            
            skway1_A=np.std(np.array(kway1_A))
            skway1_B=np.std(np.array(kway1_B))
            skway1_C=np.std(np.array(kway1_C))
            skway1_D=np.std(np.array(kway1_D))
            skway1_E=np.std(np.array(kway1_E))
            skway1_F=np.std(np.array(kway1_F))
            skway1_G=np.std(np.array(kway1_G))
            skway1_H=np.std(np.array(kway1_H))
            skway1_I=np.std(np.array(kway1_I))
            
            skway2_A=np.std(np.array(kway2_A))
            skway2_B=np.std(np.array(kway2_B))
            skway2_C=np.std(np.array(kway2_C))
            skway2_D=np.std(np.array(kway2_D))
            skway2_E=np.std(np.array(kway2_E))
            skway2_F=np.std(np.array(kway2_F))
            skway2_G=np.std(np.array(kway2_G))
            skway2_H=np.std(np.array(kway2_H))
            skway2_I=np.std(np.array(kway2_I))
            
            skway3_A=np.std(np.array(kway3_A))
            skway3_B=np.std(np.array(kway3_B))
            skway3_C=np.std(np.array(kway3_C))
            skway3_D=np.std(np.array(kway3_D))
            skway3_E=np.std(np.array(kway3_E))
            skway3_F=np.std(np.array(kway3_F))
            skway3_G=np.std(np.array(kway3_G))
            skway3_H=np.std(np.array(kway3_H))
            skway3_I=np.std(np.array(kway3_I))
            
            skway4_A=np.std(np.array(kway4_A))
            skway4_B=np.std(np.array(kway4_B))
            skway4_C=np.std(np.array(kway4_C))
            skway4_D=np.std(np.array(kway4_D))
            skway4_E=np.std(np.array(kway4_E))
            skway4_F=np.std(np.array(kway4_F))
            skway4_G=np.std(np.array(kway4_G))
            skway4_H=np.std(np.array(kway4_H))
            skway4_I=np.std(np.array(kway4_I))
            
            skway5_A=np.std(np.array(kway5_A))
            skway5_B=np.std(np.array(kway5_B))
            skway5_C=np.std(np.array(kway5_C))
            skway5_D=np.std(np.array(kway5_D))
            skway5_E=np.std(np.array(kway5_E))
            skway5_F=np.std(np.array(kway5_F))
            skway5_G=np.std(np.array(kway5_G))
            skway5_H=np.std(np.array(kway5_H))
            skway5_I=np.std(np.array(kway5_I))
            
            dA= np.array([mkway1_A,mkway2_A,mkway3_A,mkway4_A,mkway5_A])
            dB= np.array([mkway1_B,mkway2_B,mkway3_B,mkway4_B,mkway5_B])
            dC= np.array([mkway1_C,mkway2_C,mkway3_C,mkway4_C,mkway5_C])
            dD= np.array([mkway1_D,mkway2_D,mkway3_D,mkway4_D,mkway5_D])
            dE= np.array([mkway1_E,mkway2_E,mkway3_E,mkway4_E,mkway5_E])
            dF= np.array([mkway1_F,mkway2_F,mkway3_F,mkway4_F,mkway5_F])
            dG= np.array([mkway1_G,mkway2_G,mkway3_G,mkway4_G,mkway5_G])
            dH= np.array([mkway1_H,mkway2_H,mkway3_H,mkway4_H,mkway5_H])
            dI= np.array([mkway1_I,mkway2_I,mkway3_I,mkway4_I,mkway5_I])
                  
            maxval=np.hstack((dA,dB,dC,dD,dE,dF,dG,dH))
            where_are_NaNs = isnan(maxval)
            maxval[where_are_NaNs] = 0
            maxval.max()
            sA= np.array([skway1_A,skway2_A,skway3_A,skway4_A,skway5_A])
            sB= np.array([skway1_B,skway2_B,skway3_B,skway4_B,skway5_B])
            sC= np.array([skway1_C,skway2_C,skway3_C,skway4_C,skway5_C])
            sD= np.array([skway1_D,skway2_D,skway3_D,skway4_D,skway5_D])
            sE= np.array([skway1_E,skway2_E,skway3_E,skway4_E,skway5_E])
            sF= np.array([skway1_F,skway2_F,skway3_F,skway4_F,skway5_F])
            sG= np.array([skway1_G,skway2_G,skway3_G,skway4_G,skway5_G])
            sH= np.array([skway1_H,skway2_H,skway3_H,skway4_H,skway5_H])
            sI= np.array([skway1_I,skway2_I,skway3_I,skway4_I,skway5_I])
    
            y=range(len(dA))+np.ones(len(dA))
            d=np.zeros(len(dA))
            x_ticks_labels = ['','1','2','3','4','5','']
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.errorbar(y,dA,xerr=d,yerr=sA,fmt='-.ob',label='LASSO')
            ax.errorbar(y,dB,xerr=d,yerr=sB,fmt='-.or',label='Br')
            ax.errorbar(y,dC,xerr=d,yerr=sC,fmt='-.ok',label='Locop')
            ax.legend(loc='upper left',fontsize=25)
            ax.grid(True)
            ax.set_xlim(0, 4)
            ax.set_ylim(0, maxval.max()+(maxval.max()/10))
            ax.xaxis.set_major_locator(MultipleLocator(.1))
            ax.yaxis.set_major_locator(MultipleLocator(.1))
            ax.xaxis.set_minor_locator(AutoMinorLocator(10))
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax.grid(which='major', color='#CCCCCC', linestyle='--')
            ax.grid(which='minor', color='#CCCCCC', linestyle=':')
            ax.set_xticks(range(len(x_ticks_labels)))
            ax.set_xticklabels(x_ticks_labels)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Kway',fontsize=16)
            plt.ylabel('AVD Distance',fontsize=16)
            plt.tight_layout()
            plt.savefig(namedata+dondegu+str(fbp)+'/p='+str(valor)+'_h='+str(hashfun)+'_t='+str(times)+'_Hashes_4_AVD_fbp_'+str(fbp)+'_sampling_'+str(sampling)+'_.png')
            plt.close()
            Ecolumns = ["1", "2", "3",  "4", "5"]
            Erows = ["Lasso", "Br", "Locop", "Std Lasso", "Std Br", "Std Locop"]
            dfTT = pd.DataFrame(data=np.array([dA,dB,dC,sA,sB,sC]), index=Erows, columns=Ecolumns)
            dfTT.to_csv(namedata+dondegu+'p='+str(valor)+'_h='+str(hashfun)+'_Hashes_4_AVD_fbp_'+str(fbp)+'_sampling_'+str(sampling)+'_.csv', index=True)
