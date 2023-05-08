import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import os
import matplotlib.pyplot as plt
import math
from random import shuffle
from bitarray.util import pprint
from bitarray.util import ba2int
import numpy as np
import matplotlib.pyplot as plt
import mmh3
from sklearn.linear_model import Lasso
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

# df=[]
# df = pd.read_csv('Csv/votes.data')
# print(df)
# final_labels=[]
# print(df)
eliminate=[]
keepit=[]
realdata=[]
# for column in df:
#     if (df[column].dtypes=='object'):
#         keepit.append(column)
#     else:
#         eliminate.append(column)
# print(eliminate) #Attributes NO categorical



df=[]
df = pd.read_csv('csv/db_NHANES.csv')
df.columns = ['gen','age','race','edu','mar','bmi','dep','pir','qm','dia']
cols_to_drop = ['bmi','dep','pir','dia','age']



df = df.drop(cols_to_drop, axis=1)


for column in df:
    if (df[column].dtypes=='object'):
        keepit.append(column)
        array=df[column].unique()
        vals = df[column].values
        uniq, idx = np.unique(vals, return_inverse=True)
        label=[]
        label=sorted(df[column].unique(),key=str.lower)
        count=np.bincount(idx)
        aya=[]
        for i in range(0,len(label)):
            #print(uniq[i],count[i],(count[i]/len(idx)))
            aya.append(count[i]/len(idx))
        realdata.append(aya)
        df.groupby(column).count()
        # i = idx- 0.5
        # plt.figure()
        # plt.hist(i, bins=np.arange(0, idx.max()+2, 1)-.5, density=True)
        # final_labels.append(sorted(df[column].unique(),key=str.lower))
        # plt.xticks(range(0,len(sorted(df[column].unique(),key=str.lower))),sorted(df[column].unique(),key=str.lower))
        # plt.title(column)
        # plt.ylabel("Probability")
        #plt.savefig('Adults_'+column+'.png', bbox_inches='tight')
    else:
        eliminate.append(column)




attri=len(df.columns)



from bloom_aghm import BloomFilter
keepit=[]
for col in df.columns:
    keepit.append(col)

bit_ = pd.DataFrame(columns=[keepit])
hash_ = pd.DataFrame(columns=[keepit])

omega = []
m=[]
h=[]
p=1/(np.exp(4*np.log(2)*np.log(2)))#0.01Acoording to LoPub Paper
BF=[]
for (columnName, columnData) in df.iteritems():
    print("#################################################################################")
    bloomf=[]
    omega=len(df[columnName].unique())
    bloomf = BloomFilter(omega,p)
    print(columnName)
    print("Size of bit array:{}".format(bloomf.size))
    print("False positive Probability:{}".format(bloomf.p))
    print("Number of hash functions:{}".format(bloomf.hash_count))
    for reading in df[columnName].unique():
        bloomf.add(str(reading))   

    adding=[]
    adding=df[columnName]
    lbit=[]
    lhash=[]
    bf_aux=[]
    
    new=[]
    new=df[columnName].unique()
    new_hash=[]
    #print(len(new),new)
    new_hash=sorted(new,key=str.lower)

    for element in new_hash:#df[columnName].unique():
        # print(str(element))
        bloomf.check(str(element))
        # print(bloomf.bitarr)
        bf_aux.append([bloomf.bitarr[i] for i in range(len(bloomf.bitarr))])   
    case=[]
    case = {columnName: bf_aux }
    BF.append(case)
        
    
    for element in adding:

        bloomf.check(str(element))
        #bloomf.perturb(0.1)
        lhash.append([bloomf.bitarr[i] for i in range(len(bloomf.bitarr))])   

    hash_[columnName]=lhash

bit_df=[]
bit_df=bit_
df.to_csv('label_server.csv') 
hash_.to_csv('hash_server.csv') 





