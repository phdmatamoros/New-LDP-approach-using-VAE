#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:32:45 2022

@author: aghm
"""
from collections import Counter
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn import mixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from numpy import *
from itertools import product  
from sklearn.utils.extmath import cartesian
from math import log
import math
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
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
from pylab import imshow
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
#FUNCTIONS

import scipy as sp
from numpy import inf
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ARDRegression, LinearRegression, BayesianRidge,LassoCV,RidgeCV
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler




def copula_gaussian(n, correlation,kwayr):
    zl=[]
    for w in range(kwayr):
        zl.append(sp.stats.norm.rvs(loc=0, scale=1, size=n,))
    Z = np.matrix(zl)
    # Construct the correlation matrix and Cholesky Decomposition
    rho = correlation
    cholesky = np.linalg.cholesky(rho)
    # Apply Cholesky and extract X and Y
    Z_XY = cholesky * Z
    val=[]
    acdf=[]
    for w in range(kwayr):
        val.append(np.array(Z_XY[w,:]).flatten())
        acdf.append(sp.stats.norm.cdf(np.array(Z_XY[w,:]).flatten(),loc=0, scale=1))
        #acdf.append(sp.stats.norm.pdf(np.array(Z_XY[w,:]).flatten(),loc=0, scale=1))
    return val,acdf 


def copula_gaussian_t(n, correlation,kwayr,freedoom):
    
    ''' Student's t distributed random variates t_X and t_Y with correlation 'p'
        and degrees of freedom 'df'
    '''
    
    # Gaussian Copula
    zl=[]
    for w in range(kwayr):
        zl.append(sp.stats.norm.rvs(loc=0, scale=1, size=n,))
    Z = np.matrix(zl)
    # Construct the correlation matrix and Cholesky Decomposition
    rho = correlation
    cholesky = np.linalg.cholesky(rho)
    # Apply Cholesky and extract X and Y
    Z_XY = cholesky * Z
    val=[]
    acdf=[]
    for w in range(kwayr):
        val.append(np.array(Z_XY[w,:]).flatten())
        acdf.append(sp.stats.norm.cdf(np.array(Z_XY[w,:]).flatten(),loc=0, scale=1))
        #acdf.append(sp.stats.norm.pdf(np.array(Z_XY[w,:]).flatten(),loc=0, scale=1))
    
    # Chi Squared Sample
    ChiSquared = np.random.chisquare(df=freedoom, size=n)

    # Stident's t distributed random variables
    st=[]
    stcdf=[]
    for w in range(kwayr):
        st.append(val[w]/ (np.sqrt(ChiSquared / freedoom)))
        stcdf.append(sp.stats.t.cdf(st[w], df=freedoom, loc=0, scale=1))

    
    return st,stcdf

from scipy.special import rel_entr

def kl_divergence(a, b):
	return sum(a[i] * np.log(a[i]/b[i]) for i in range(len(a)))

def perturb_aghm(bit,fbp):
     #secretsGen=secrets.SystemRandom()
     p_sample=random.uniform(0, 1)#secretsGen.randint(0,100000)/100000
     sample=bit
     if p_sample < fbp:############################################################################################CHANGEEEE
         sample=random.choice([0,1])
     return sample
 
def perturb_aghmhexal(bit,fbp):
     #secretsGen=secrets.SystemRandom()
     p_sample=random.uniform(0, 1)#secretsGen.randint(0,100000)/100000
     sample=bit
     if p_sample+.0000001 < fbp:############################################################################################CHANGEEEE
         sample=random.choice(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'])
     return sample
 
def str2vec(aux):
	v = []
	for s in aux:
		if(s in ['0', '1']): v.append(int(s))
	return(v)

def str2vechex(aux):
	v = []
	for s in aux:
		if(s in ['0', '1']): v.append(s)
	return(v)

def vector_aghm(bitmap,N,fbp):
    bmap=[]
    for ele in bitmap:
        bmap.append((ele-(fbp*N/2))/(1-fbp))
    return(np.array(bmap))


def vector_aghmv2(bitmap,N,fbp):
    bmap=[]
    for ele in bitmap:
        fb=random.choice([0,fbp,0.99])
        bmap.append((ele-(fb*N/2))/(1-fb))
    return(np.array(bmap))


def df2np(bit2):
	bit3 = []
	for i in range(bit2.shape[0]):
		v = []
		for j in range(bit2.shape[1]):
			#print(i,j)
			v +=  str2vec(bit2.iloc[i,j])
		bit3.append(np.array(v, dtype=int))
	bit4 = np.array(bit3, dtype=int)
	return(bit4)

def df2nperturb(bit2,fbp):
    bit3 = []
    for i in range(bit2.shape[0]):
        v=[]
        for j in range(bit2.shape[1]):
            v += str2vec(bit2.iloc[i,j])
            # v=(list(map(perturb_aghm, v,fbp*np.ones(len(bit2)))))
        bit3.append(np.array(v, dtype=int))
    bit4 = np.array(bit3, dtype=int)
    return(bit4)


def df2nperturbx2(bit2,fbp):
    bit3 = []
    for i in range(bit2.shape[0]):
        v=[]
        for j in range(bit2.shape[1]):
            v += str2vec(bit2.iloc[i,j])
        v=np.hstack((v,v))#v[::-1]))
        v=(list(map(perturb_aghm, v,fbp*np.ones(len(bit2)))))
        bit3.append(np.array(v, dtype=int))
    bit4 = np.array(bit3, dtype=int)
    return(bit4)




hexal_index={}
hexal_index['{}'.format('0')]='0000'
hexal_index['{}'.format('1')]='0001'
hexal_index['{}'.format('2')]='0010'
hexal_index['{}'.format('3')]='0011'
hexal_index['{}'.format('4')]='0100'
hexal_index['{}'.format('5')]='0101'
hexal_index['{}'.format('6')]='0110'
hexal_index['{}'.format('7')]='0111'
hexal_index['{}'.format('8')]='1000'
hexal_index['{}'.format('9')]='1001'
hexal_index['{}'.format('A')]='1010'
hexal_index['{}'.format('B')]='1011'
hexal_index['{}'.format('C')]='1100'
hexal_index['{}'.format('D')]='1101'
hexal_index['{}'.format('E')]='1110'
hexal_index['{}'.format('F')]='1111'


def df2hexaltobin(bit2):
    bit3=[]
    for i in range(len(bit2)):
        v=[]
        aux=bit2[i]
        for j in aux:
            v += hexal_index[j]
            #v=(list(map(perturb_aghmhexal, v, fbp*ones(len(bit2)))))
        v=''.join(str(i) for i in v)
        bit3.append(v)
    return(bit3)


def df2nperturbhexal(bit2,fbp):
    bit3 = []
    for i in range(len(bit2)):
        v=[]
        aux=bit2[i]
        for j in aux:
            v += j
            v=(list(map(perturb_aghmhexal, v, fbp*ones(len(bit2)))))
        bit3.append(''.join(str(i) for i in v))
    return(bit3)


def binToHexa(n):
    bnum = int(n)
    temp = 0
    mul = 1
      
    # counter to check group of 4
    count = 1
      
    # char array to store hexadecimal number
    hexaDeciNum = ['0'] * 100
      
    # counter for hexadecimal number array
    i = 0
    while bnum != 0:
        rem = bnum % 10
        temp = temp + (rem*mul)
          
        # check if group of 4 completed
        if count % 4 == 0:
            
            # check if temp < 10
            if temp < 10:
                hexaDeciNum[i] = chr(temp+48)
            else:
                hexaDeciNum[i] = chr(temp+55)
            mul = 1
            temp = 0
            count = 1
            i = i+1
              
        # group of 4 is not completed
        else:
            mul = mul*2
            count = count+1
        bnum = int(bnum/10)
          
    # check if at end the group of 4 is not
    # completed
    if count != 1:
        hexaDeciNum[i] = chr(temp+48)
          
    # check at end the group of 4 is completed
    if count == 1:
        i = i-1
          
    # printing hexadecimal number
    # array in reverse order
    out=hexaDeciNum[0]
    return(out)

def dftohexal(bit2,fbp):
    bit3 = []
    
    for i in range(bit2.shape[0]):
        v=[]
        bit=[]
        for j in range(bit2.shape[1]):
            v += str2vechex(bit2.iloc[i,j])
        for a, b, c, d in zip(*[iter(v)]*4):
            aux=''.join(str(i) for i in [a,b,c,d])
            bit.append(str(binToHexa(aux)))
    
        bit3.append(''.join(str(i) for i in bit))
    return(bit3)

def df2size(bit2):
    bit3 = [1]
    for i in range(len(bit2.columns)):
        v= str2vec(bit2.iloc[0,i])
        bit3.append(len(v))
    return(bit3)

def df2sizex2(bit2):
    bit3 = [1]
    for i in range(len(bit2.columns)):
        v= str2vec(bit2.iloc[0,i])
        bit3.append(2*len(v))
    return(bit3)

def hexalfinal(bit2):
    bit3 = []
    for i in range(len(bit2)):
        v=[]
        v += str2vec(bit2[i])
        bit3.append(np.array(v, dtype=int))
    bit4 = np.array(bit3, dtype=int)
    return(bit4)

def df2VAEperturb(bit2,fbp):
    bit3 = []
    for i in range(bit2.shape[0]):
        v=[]
        v += str2vec(bit2.iloc[i])
        v=(list(map(perturb_aghm, v)))
        bit3.append(np.array(v, dtype=int))
    bit4 = np.array(bit3, dtype=int)
    return(bit4)

#d2
#attributes
#dict_perturb used to calculated Y
#kway
#bit
#fbp
#value_cat 
#df
def LASSO_original(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,BloomFilter,valor,hashfun):
    # print('LASSO')
    dfmulQ=[]
    ant=[]

    ii=[]
    tuples=[]
    index=[]
    dfmul=[]
    # print('att',attributes)
    for poi in attributes:
        # print('poi',poi)
        # print('value_cat[poi]',value_cat[poi])
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

    
    listb=[]
    for v in dfmulQ:
        if (None in v) is False:
            listb.append(v)        
    # valor=.022


    ant = ant.astype('int')
    f=1
    M=[]
    for col in ant.columns:
        # print(col)
        omega=len(df[col].unique())
        val_m=math.ceil(omega*log(1/valor)/(log(2)*log(2)))
        
        rappor=[]
        rappor = RAPPORClient(f=0, m=val_m, hash_funcs=generate_hash_funcs(hashfun,val_m))
        vector=np.array(ant[col])
        coding = []
        for ele in vector:
            coding.append(rappor.privatise(ele))
        if f==1:
            M=np.array(coding)
            f=f+1
        else:
            M=np.hstack((M,np.array(coding)))


    df2=[]
    df2=d2[attributes]
    bit4=[]
    for ele,ind in  zip(attributes, range(len(attributes))):
        if ind==0:
            bit4=dict_perturb[ele]
        if ind!=0:
            bit4=np.hstack((bit4,dict_perturb[ele]))

    Y=[]
    Y = np.sum(bit4, axis = 0)

    
    Y = vector_aghm(Y,len(bit4),fbp)
    

    clf=[]
    clf = Lasso(alpha = 0.1)#original 0.1
    clf.fit(M.T, Y)
    p_lasso=[]
    coef=abs(clf.coef_)#/len(bit4)
    p_lasso=coef/(sum(coef))

    p_lasso=pd.Series(list(p_lasso), index=index)
    p_lasso=p_lasso.to_frame()
    p_lasso.columns = ["Lasso"]
    return p_lasso, coef/(sum(coef))


def Br_original(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,BloomFilter,valor,hashfun):
    # print('LASSO')
    dfmulQ=[]
    ant=[]

    ii=[]
    tuples=[]
    index=[]
    dfmul=[]
    # print('att',attributes)
    for poi in attributes:
        # print('poi',poi)
        # print('value_cat[poi]',value_cat[poi])
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

    listb=[]
    for v in dfmulQ:
        if (None in v) is False:
            listb.append(v)        
    # valor=.022
    # print(valor)

    ant = ant.astype('int')
    f=1
    M=[]
    for col in ant.columns:
        # print(col)
        omega=len(df[col].unique())
        val_m=math.ceil(omega*log(1/valor)/(log(2)*log(2)))
        rappor=[]
        rappor = RAPPORClient(f=0, m=val_m, hash_funcs=generate_hash_funcs(hashfun,val_m))
        vector=np.array(ant[col])
        coding = []
        for ele in vector:
            coding.append(rappor.privatise(ele))
        if f==1:
            M=np.array(coding)
            f=f+1
        else:
            M=np.hstack((M,np.array(coding)))


    df2=[]
    df2=d2[attributes]
    bit4=[]
    for ele,ind in  zip(attributes, range(len(attributes))):
        if ind==0:
            bit4=dict_perturb[ele]
        if ind!=0:
            bit4=np.hstack((bit4,dict_perturb[ele]))



    Y=[]
    Y = np.sum(bit4, axis = 0)
    Y = vector_aghm(Y,len(bit4),fbp)


    clf=[]
    clf=BayesianRidge(compute_score=True, n_iter=30)
    clf.fit(M.T, Y)
    p_lasso=[]
    coef=abs(clf.coef_)#/len(bit4)
    p_lasso=coef/(sum(coef))
    
    p_lasso=pd.Series(list(p_lasso), index=index)
    p_lasso=p_lasso.to_frame()
    p_lasso.columns = ["Br"]
    return p_lasso, coef/(sum(coef))



def method(d2, attributes,dict_perturb,kway,bit,fbp,value_cat,df,BloomFilter,valor,hashfun):
    # print('LASSO')
    dfmulQ=[]
    ant=[]

    ii=[]
    tuples=[]
    index=[]
    dfmul=[]
    indexmat=[]
    # print('att',attributes)
    for poi in attributes:
        # print('poi',poi)
        # print('value_cat[poi]',value_cat[poi])
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
        indexmat=[ii[:,0]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
     
        
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0]]
        ant = ant.astype('int')

        
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
        indexmat=[ii[:,0],ii[:,1]]
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
        indexmat=[ii[:,0],ii[:,1],ii[:,2]]
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
        indexmat=[ii[:,0],ii[:,1],ii[:,2],ii[:,3]]
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
        indexmat=[ii[:,0],ii[:,1],ii[:,2],ii[:,3],ii[:,4]]
        tuples = list(zip(*tuples))
        index = pd.MultiIndex.from_tuples(tuples, names=attributes)
        ant=pd.DataFrame(lista) 
        ant.columns = [attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]]
        ant = ant.astype('int')

    listb=[]
    for v in dfmulQ:
        if (None in v) is False:
            listb.append(v)        
    # valor=.022
    # print(valor)

    ant = ant.astype('int')
    f=1
    M=[]
    for col in ant.columns:
        # print(col)
        omega=len(df[col].unique())
        val_m=math.ceil(omega*log(1/valor)/(log(2)*log(2)))
        rappor=[]
        rappor = RAPPORClient(f=0, m=val_m, hash_funcs=generate_hash_funcs(hashfun,val_m))
        vector=np.array(ant[col])
        coding = []
        for ele in vector:
            # print(ele)
            coding.append(rappor.privatise(ele))
        if f==1:
            M=np.array(coding)
            f=f+1
        else:
            M=np.hstack((M,np.array(coding)))


    df2=[]
    df2=d2[attributes]
    bit4=[]
    for ele,ind in  zip(attributes, range(len(attributes))):
        if ind==0:
            bit4=dict_perturb[ele]
            tam=len(bit4)
        if ind!=0:
            bit4=np.hstack((bit4,dict_perturb[ele]))


        
    # for changing in range(0, 9, 1):
    Y = np.sum(bit4, axis = 0)
        # Y = vector_aghm(Y,len(bit4))
    Y = vector_aghmv2(Y,len(bit4),fbp)
    clf=[]
    clf =  BayesianRidge(compute_score=True, n_iter=1000)
    clf.fit(M.T, Y)
    p_lasso=[]
    p_lasso=np.array(abs(clf.coef_)/sum(abs(clf.coef_)))
                         
   
    # 1/0
    # cc=0
    # Ma=M
    # for p in p_lasso:
    #     Ma[cc,:]=int(len(bit4)*p)*M[cc,:]
    #     cc+=1
    
    
    # akb=(np.sum(Ma, axis = 0))
    # akb=(akb/sum(akb))
    # aka=Y
    # aka[aka>0]=0
    # aka=(aka/sum(aka))
 
    # # plt.plot(akb)
    # # plt.plot(aka)

    # clf.fit(M.T, aka)
    # p_lasso=[]
    # p_lasso=abs(clf.coef_)/sum(abs(clf.coef_))
    # print(p_lasso)
    
    
    
    # 1/0
    # for p in p_lasso:
    #     Ma[cc,:]=int(len(bit4)*p)*M[cc,:]
    #     cc+=1

    p_lasso=pd.Series(list(p_lasso), index=index)    
    p_lasso=p_lasso.to_frame()    
    p_lasso.columns = ["Method"]

    return p_lasso, Y



