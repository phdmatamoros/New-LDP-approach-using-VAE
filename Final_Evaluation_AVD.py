import pandas as pd
import numpy as np
import random 
import secrets
#from bloom_aghm import BloomFilter
from sklearn.linear_model import Lasso
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import matplotlib.ticker as mtick
warnings.filterwarnings('ignore')
from sklearn.linear_model import  BayesianRidge

import gc


 

def perturb_aghm(bit):
     secretsGen=secrets.SystemRandom()
     p_sample=secretsGen.randint(0,100000)/100000
     sample=bit
     if p_sample <= fbp:############################################################################################CHANGEEEE
         sample=random.choice([0, 1])
     return sample
 
    
 
def str2vec(aux):
	v = []
	for s in aux:
		if(s in ['0', '1']): v.append(int(s))
	return(v)

def vector_aghm(bitmap):
    bmap=[]
    for ele in bitmap:
        bmap.append((ele-(fbp*(N/2)))/(1-fbp))
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
            #v=(list(map(perturb_aghm, v)))
        bit3.append(np.array(v, dtype=int))
    bit4 = np.array(bit3, dtype=int)
    return(bit4)
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

df = pd.read_csv('label_server.csv')
df=df.loc[:, ~df.columns.str.contains('^Unnamed')]

dk = pd.read_csv('hash_server.csv')
dk=dk.loc[:, ~dk.columns.str.contains('^Unnamed')]




unique_bin={}
for col in dk.columns:
    inside={}
    for i,j in zip((df[col]).unique(),(dk[col]).unique()):
        inside['{}'.format(i)]=j
    unique_bin['{}'.format(col)]=inside


fbp_array=[0.3,0.5,0.7,0.9]
kways=[2,3,4]






L_ktw_A=[]
L_ktw_B=[]
L_ktw_C=[]
L_ktw_D=[]

L_kt_A=[]
L_kt_B=[]
L_kt_C=[]
L_kt_D=[]

L_kf_A=[]
L_kf_B=[]
L_kf_C=[]
L_kf_D=[]


V_ktw_A=[]
V_ktw_B=[]
V_ktw_C=[]
V_ktw_D=[]

V_kt_A=[]
V_kt_B=[]
V_kt_C=[]
V_kt_D=[]

V_kf_A=[]
V_kf_B=[]
V_kf_C=[]
V_kf_D=[]

B_ktw_A=[]
B_ktw_B=[]
B_ktw_C=[]
B_ktw_D=[]

B_kt_A=[]
B_kt_B=[]
B_kt_C=[]
B_kt_D=[]

B_kf_A=[]
B_kf_B=[]
B_kf_C=[]
B_kf_D=[]


#############
print("Working")
for times in range(0,100):
    print(times)
    indexa=[]
    space=list(np.linspace(0,len(df), num=10).astype(int))
    space.pop(0)
    aux2=np.zeros((4, 1+len(space)))
    aux2[0,0]=0.3
    aux2[1,0]=0.5
    aux2[2,0]=0.7
    aux2[3,0]=0.9

    Lk2 = pd.DataFrame(aux2,columns=[np.hstack(('fbp',space))])
    Lk3 = pd.DataFrame(aux2,columns=[np.hstack(('fbp',space))])
    Lk4 = pd.DataFrame(aux2,columns=[np.hstack(('fbp',space))])
    Vk2 = pd.DataFrame(aux2,columns=[np.hstack(('fbp',space))])
    Vk3 = pd.DataFrame(aux2,columns=[np.hstack(('fbp',space))])
    Vk4 = pd.DataFrame(aux2,columns=[np.hstack(('fbp',space))])
    Bk2 = pd.DataFrame(aux2,columns=[np.hstack(('fbp',space))])
    Bk3 = pd.DataFrame(aux2,columns=[np.hstack(('fbp',space))])
    Bk4 = pd.DataFrame(aux2,columns=[np.hstack(('fbp',space))])

    Lk2=Lk2.set_index(Lk2.iloc[:, 0].name)
    Lk3=Lk3.set_index(Lk3.iloc[:, 0].name)
    Lk4=Lk4.set_index(Lk4.iloc[:, 0].name)
    Vk2=Vk2.set_index(Vk2.iloc[:, 0].name)
    Vk3=Vk3.set_index(Vk3.iloc[:, 0].name)
    Vk4=Vk4.set_index(Vk4.iloc[:, 0].name)
    Bk2=Bk2.set_index(Bk2.iloc[:, 0].name)
    Bk3=Bk3.set_index(Bk3.iloc[:, 0].name)
    Bk4=Bk4.set_index(Bk4.iloc[:, 0].name)
    
    columns=-1
    for num in space:
        columns+=1
        indexa.append(num)
        N=num
        for kway in kways:
            fbp_kikn={}
            fbp_avd={}
            fbp_max={}
            rows=-1
            for fbp in fbp_array:
                rows+=1
                kikn_lasso=[]
                kikn_vae=[]
                avd_lasso=[]
                avd_vae=[]
                max_lasso=[]
                max_vae=[]
                random.seed(times)#1
                attributes=random.sample(list(df.columns), kway)
                #print("U->",num," ",attributes," choosen", "fbp->", fbp)
                vae_fs=[]
                vae_cen=[]
                vae_fs =pd.read_csv('csv_eval/VAE_FULLSPACE_flopprob_'+str(fbp)+'_.csv')
    
                df2 = df[attributes].head(N)
                fq = df2.value_counts()
                
                df3 = vae_fs[attributes].head(N)
                fq3 = df3.value_counts()

        
                ################################
                bit=[]
                bit2=[]
                bit4=[]
                
                dp = pd.read_csv('csv_eval/perturbed_dataset_'+str(fbp)+'_.csv')
                dp=  dp.loc[:, ~dp.columns.str.contains('^Unnamed')]
                
                bit = pd.read_csv('hash_server.csv')
                bit=bit.loc[:, ~bit.columns.str.contains('^Unnamed')]
                bit2 = bit[attributes].head(N)
                
                
                bit4 = dp[attributes].head(N)#df2nperturb(bit2,fbp)
                bit4 = df2nperturb(bit4,fbp)
    
                ###################
                Y = np.sum(bit4, axis = 0)
            
                Y    =vector_aghm(Y)
           
                candidate = []
                for v in fq.index:
                        b = bit2[(df2 == v).all(axis = 1)]
                        if kway==2:
                            bc = np.array(str2vec(b.iloc[0,0]) + str2vec(b.iloc[0,1]))
                        if kway==3:
                            bc = np.array(str2vec(b.iloc[0,0]) + str2vec(b.iloc[0,1])+ str2vec(b.iloc[0,2]))
                        if kway==4:
                            bc = np.array(str2vec(b.iloc[0,0]) + str2vec(b.iloc[0,1])+ str2vec(b.iloc[0,2])+ str2vec(b.iloc[0,3]))
                        candidate.append(bc)
                M = np.array(candidate)
                ########################
                clf = Lasso(alpha = 0.1, fit_intercept=False,)#max_iter=5000)
                clf.fit(M.T, Y)
                p_est = abs(clf.coef_)/sum(abs(clf.coef_))
                
                
                # clf=[]
                # clf=BayesianRidge(compute_score=True, n_iter=30).fit(M.T, Y)
                # p_br=[]
                # coef=abs(clf.coef_)#/len(bit4)
                # p_br=coef/(sum(coef))
                

    
                ########################
                p_true = fq/fq.sum()
                p_vae= fq3/fq3.sum()
    

    
                df2 = pd.DataFrame()
                df2['ORIGINAL']=p_true
                df2['LASSO'] = p_est
                df2['VAE'] = p_vae
                df3=df2.replace(r'^\s*$', 0, regex=True)
                df3=df3.fillna(0)
    
            

                avd_lasso= 0.5*(df3['LASSO'] - df3['ORIGINAL']).abs().sum()
                avd_vae= 0.5*(df3['VAE'] - df3['ORIGINAL']).abs().sum()                
                avd_br= 0               


                if  kway==2:
                    Lk2.iat[rows,columns]=avd_lasso
                    Vk2.iat[rows,columns]=avd_vae
                    Bk2.iat[rows,columns]=avd_br
                if  kway==3:
                    Lk3.iat[rows,columns]=avd_lasso
                    Vk3.iat[rows,columns]=avd_vae
                    Bk3.iat[rows,columns]=avd_br
                if  kway==4:
                    Lk4.iat[rows,columns]=avd_lasso
                    Vk4.iat[rows,columns]=avd_vae
                    Bk4.iat[rows,columns]=avd_br
         

    L_ktw_A.append(np.array(Lk2.iloc[0]))
    L_ktw_B.append(np.array(Lk2.iloc[1]))
    L_ktw_C.append(np.array(Lk2.iloc[2]))
    L_ktw_D.append(np.array(Lk2.iloc[3]))

    L_kt_A.append(np.array(Lk3.iloc[0]))
    L_kt_B.append(np.array(Lk3.iloc[1]))
    L_kt_C.append(np.array(Lk3.iloc[2]))
    L_kt_D.append(np.array(Lk3.iloc[3]))

    L_kf_A.append(np.array(Lk4.iloc[0]))
    L_kf_B.append(np.array(Lk4.iloc[1]))
    L_kf_C.append(np.array(Lk4.iloc[2]))
    L_kf_D.append(np.array(Lk4.iloc[3]))


    V_ktw_A.append(np.array(Vk2.iloc[0]))
    V_ktw_B.append(np.array(Vk2.iloc[1]))
    V_ktw_C.append(np.array(Vk2.iloc[2]))
    V_ktw_D.append(np.array(Vk2.iloc[3]))

    V_kt_A.append(np.array(Vk3.iloc[0]))
    V_kt_B.append(np.array(Vk3.iloc[1]))
    V_kt_C.append(np.array(Vk3.iloc[2]))
    V_kt_D.append(np.array(Vk3.iloc[3]))

    V_kf_A.append(np.array(Vk4.iloc[0]))
    V_kf_B.append(np.array(Vk4.iloc[1]))
    V_kf_C.append(np.array(Vk4.iloc[2]))
    V_kf_D.append(np.array(Vk4.iloc[3]))
    
    B_ktw_A.append(np.array(Bk2.iloc[0]))
    B_ktw_B.append(np.array(Bk2.iloc[1]))
    B_ktw_C.append(np.array(Bk2.iloc[2]))
    B_ktw_D.append(np.array(Bk2.iloc[3]))

    B_kt_A.append(np.array(Bk3.iloc[0]))
    B_kt_B.append(np.array(Bk3.iloc[1]))
    B_kt_C.append(np.array(Bk3.iloc[2]))
    B_kt_D.append(np.array(Bk3.iloc[3]))

    B_kf_A.append(np.array(Bk4.iloc[0]))
    B_kf_B.append(np.array(Bk4.iloc[1]))
    B_kf_C.append(np.array(Bk4.iloc[2]))
    B_kf_D.append(np.array(Bk4.iloc[3]))
    
    ###########################################################################
    ###########################################################################
    # L_ktwA=np.array(L_ktw_A)
    # L_ktwB=np.array(L_ktw_B)
    # L_ktwC=np.array(L_ktw_C)
    # L_ktwD=np.array(L_ktw_D)
    
    # L_ktA=np.array(L_kt_A)
    # L_ktB=np.array(L_kt_B)
    # L_ktC=np.array(L_kt_C)
    # L_ktD=np.array(L_kt_D)
    
    # L_kfA=np.array(L_kf_A)
    # L_kfB=np.array(L_kf_B)
    # L_kfC=np.array(L_kf_C)
    # L_kfD=np.array(L_kf_D)
    
    # V_ktwA=np.array(V_ktw_A)
    # V_ktwB=np.array(V_ktw_B)
    # V_ktwC=np.array(V_ktw_C)
    # V_ktwD=np.array(V_ktw_D)
    
    # V_ktA=np.array(V_kt_A)
    # V_ktB=np.array(V_kt_B)
    # V_ktC=np.array(V_kt_C)
    # V_ktD=np.array(V_kt_D)
    
    # V_kfA=np.array(V_kf_A)
    # V_kfB=np.array(V_kf_B)
    # V_kfC=np.array(V_kf_C)
    # V_kfD=np.array(V_kf_D)
    
    
    # B_ktwA=np.array(B_ktw_A)
    # B_ktwB=np.array(B_ktw_B)
    # B_ktwC=np.array(B_ktw_C)
    # B_ktwD=np.array(B_ktw_D)
    
    # B_ktA=np.array(B_kt_A)
    # B_ktB=np.array(B_kt_B)
    # B_ktC=np.array(B_kt_C)
    # B_ktD=np.array(B_kt_D)
    
    # B_kfA=np.array(B_kf_A)
    # B_kfB=np.array(B_kf_B)
    # B_kfC=np.array(B_kf_C)
    # B_kfD=np.array(B_kf_D)
    
    # ######################
    
    
    
    # ####0.1
    # dL = []
    # dV = []
    # sL = []
    # sV = []
    # dfTT = []
    # dL =np.array([np.mean(L_ktwA,0)[-1],np.mean(L_ktA,0)[-1],np.mean(L_kfA,0)[-1]])
    # dV =np.array([np.mean(V_ktwA,0)[-1],np.mean(V_ktA,0)[-1],np.mean(V_kfA,0)[-1]])
    # sL =np.array([np.std(L_ktwA,0)[-1],np.std(L_ktA,0)[-1],np.std(L_kfA,0)[-1]])
    # sV =np.array([np.std(V_ktwA,0)[-1],np.std(V_ktA,0)[-1],np.std(V_kfA,0)[-1]])
    # Ecolumns = ["2", "3",  "4"]
    # Erows = ["Lasso",  "Std Lasso", "VAE", "Std VAE"]
    # dfTT = []
    # dfTT = pd.DataFrame(data=np.array([dL,sL,dV,sV]), index=Erows, columns=Ecolumns)
    # dfTT.to_csv('0.3_.csv', index=True)
    
    # ####0.4
    # dL = []
    # dV = []
    # sL = []
    # sV = []
    # dfTT = []
    # dL =np.array([np.mean(L_ktwB,0)[-1],np.mean(L_ktB,0)[-1],np.mean(L_kfB,0)[-1]])
    # dV =np.array([np.mean(V_ktwB,0)[-1],np.mean(V_ktB,0)[-1],np.mean(V_kfB,0)[-1]])
    # sL =np.array([np.std(L_ktwB,0)[-1],np.std(L_ktB,0)[-1],np.std(L_kfB,0)[-1]])
    # sV =np.array([np.std(V_ktwB,0)[-1],np.std(V_ktB,0)[-1],np.std(V_kfB,0)[-1]])
    # Ecolumns = ["2", "3",  "4"]
    # Erows = ["Lasso",  "Std Lasso", "VAE", "Std VAE"]
    # dfTT = []
    # dfTT = pd.DataFrame(data=np.array([dL,sL,dV,sV]), index=Erows, columns=Ecolumns)
    # dfTT.to_csv('0.5_.csv', index=True)
    # ####0.7
    # dL = []
    # dV = []
    # sL = []
    # sV = []
    # dfTT = []
    # dL =np.array([np.mean(L_ktwC,0)[-1],np.mean(L_ktC,0)[-1],np.mean(L_kfC,0)[-1]])
    # dV =np.array([np.mean(V_ktwC,0)[-1],np.mean(V_ktC,0)[-1],np.mean(V_kfC,0)[-1]])
    # sL =np.array([np.std(L_ktwC,0)[-1],np.std(L_ktC,0)[-1],np.std(L_kfC,0)[-1]])
    # sV =np.array([np.std(V_ktwC,0)[-1],np.std(V_ktC,0)[-1],np.std(V_kfC,0)[-1]])
    # Ecolumns = ["2", "3",  "4"]
    # Erows = ["Lasso",  "Std Lasso", "VAE", "Std VAE"]
    # dfTT = []
    # dfTT = pd.DataFrame(data=np.array([dL,sL,dV,sV]), index=Erows, columns=Ecolumns)
    # dfTT.to_csv('0.7_.csv', index=True)
    # ####0.9
    # dL = []
    # dV = []
    # sL = []
    # sV = []
    # dfTT = []
    # dL =np.array([np.mean(L_ktwD,0)[-1],np.mean(L_ktD,0)[-1],np.mean(L_kfD,0)[-1]])
    # dV =np.array([np.mean(V_ktwD,0)[-1],np.mean(V_ktD,0)[-1],np.mean(V_kfD,0)[-1]])
    # sL =np.array([np.std(L_ktwD,0)[-1],np.std(L_ktD,0)[-1],np.std(L_kfD,0)[-1]])
    # sV =np.array([np.std(V_ktwD,0)[-1],np.std(V_ktD,0)[-1],np.std(V_kfD,0)[-1]])
    # Ecolumns = ["2", "3",  "4"]
    # Erows = ["Lasso",  "Std Lasso", "VAE", "Std VAE"]
    # dfTT = []
    # dfTT = pd.DataFrame(data=np.array([dL,sL,dV,sV]), index=Erows, columns=Ecolumns)
    # dfTT.to_csv('0.9_.csv', index=True)
    
    
    
    
    # plt.plot(space,np.mean(L_ktwA,0),linestyle='dashed',c='blue',label = "LASSO-kway=2") 
    # plt.plot(space,np.mean(L_ktA,0),linestyle='dashed',c='orange',label = "LASSO-kway=3") 
    # plt.plot(space,np.mean(L_kfA,0),linestyle='dashed',c='green',label = "LASSO-kway=4") 
    
    # plt.plot(space,np.mean(V_ktwA,0),c='blue',label = "VAE kway=2") 
    # plt.plot(space,np.mean(V_ktA,0),c='orange',label = "VAE kway=3") 
    # plt.plot(space,np.mean(V_kfA,0),c='green',label = "VAE kway=4") 
    
    # # plt.plot(space,np.mean(B_ktw_A,0),linestyle='dashdot',c='blue',label = "BRR kway=2") 
    # # plt.plot(space,np.mean(B_kt_A,0),linestyle='dashdot',c='orange',label = "BRR kway=3") 
    # # plt.plot(space,np.mean(B_kf_A,0),linestyle='dashdot',c='green',label = "BRR kway=4") 
    
    # plt.ylim(0,1)
    # plt.legend()
    # plt.xlabel("Users", fontsize=16)
    # plt.ylabel("AVD", fontsize=16)
    # plt.grid(True)
    # plt.tight_layout()
    # chin='Plots/0.3.jpg' 
    # plt.savefig(chin)
    # plt.close()     
    # ###########
    # plt.plot(space,np.mean(L_ktwB,0),linestyle='dashed',c='blue',label = "LASSO-kway=2") 
    # plt.plot(space,np.mean(L_ktB,0),linestyle='dashed',c='orange',label = "LASSO-kway=3") 
    # plt.plot(space,np.mean(L_kfB,0),linestyle='dashed',c='green',label = "LASSO-kway=4") 
    
    # plt.plot(space,np.mean(V_ktwB,0),c='blue',label = "VAE kway=2") 
    # plt.plot(space,np.mean(V_ktB,0),c='orange',label = "VAE kway=3") 
    # plt.plot(space,np.mean(V_kfB,0),c='green',label = "VAE kway=4") 
    
    # # plt.plot(space,np.mean(B_ktw_B,0),linestyle='dashdot',c='blue',label = "BRR kway=2") 
    # # plt.plot(space,np.mean(B_kt_B,0),linestyle='dashdot',c='orange',label = "BRR kway=3") 
    # # plt.plot(space,np.mean(B_kf_B,0),linestyle='dashdot',c='green',label = "BRR kway=4") 
    # plt.ylim(0,1)
    # plt.legend()
    # plt.xlabel("Users", fontsize=16)
    # plt.ylabel("AVD", fontsize=16)
    # plt.grid(True)
    # plt.tight_layout()
    # chin='Plots/0.5.jpg' 
    # plt.savefig(chin)
    # plt.close()     
    # #########
    # plt.plot(space,np.mean(L_ktwC,0),linestyle='dashed',c='blue',label = "LASSO-kway=2") 
    # plt.plot(space,np.mean(L_ktC,0),linestyle='dashed',c='orange',label = "LASSO-kway=3") 
    # plt.plot(space,np.mean(L_kfC,0),linestyle='dashed',c='green',label = "LASSO-kway=4") 
    
    # plt.plot(space,np.mean(V_ktwC,0),c='blue',label = "VAE kway=2") 
    # plt.plot(space,np.mean(V_ktC,0),c='orange',label = "VAE kway=3") 
    # plt.plot(space,np.mean(V_kfC,0),c='green',label = "VAE kway=4")
    
    # # plt.plot(space,np.mean(B_ktw_C,0),linestyle='dashdot',c='blue',label = "BRR kway=2") 
    # # plt.plot(space,np.mean(B_kt_C,0),linestyle='dashdot',c='orange',label = "BRR kway=3") 
    # # plt.plot(space,np.mean(B_kf_C,0),linestyle='dashdot',c='green',label = "BRR kway=4") 
    # plt.ylim(0,1)
    # plt.legend()
    # plt.xlabel("Users", fontsize=16)
    # plt.ylabel("AVD", fontsize=16)
    # plt.grid(True)
    # plt.tight_layout()
    # chin='Plots/0.7.jpg' 
    # plt.savefig(chin)
    # plt.close()     
    # #########
    # plt.plot(space,np.mean(L_ktwD,0),linestyle='dashed',c='blue',label = "LASSO-kway=2") 
    # plt.plot(space,np.mean(L_ktD,0),linestyle='dashed',c='orange',label = "LASSO-kway=3") 
    # plt.plot(space,np.mean(L_kfD,0),linestyle='dashed',c='green',label = "LASSO-kway=4") 
    
    # plt.plot(space,np.mean(V_ktwD,0),c='blue',label = "VAE kway=2") 
    # plt.plot(space,np.mean(V_ktD,0),c='orange',label = "VAE kway=3") 
    # plt.plot(space,np.mean(V_kfD,0),c='green',label = "VAE kway=4") 
    
    # # plt.plot(space,np.mean(B_ktw_D,0),linestyle='dashdot',c='blue',label = "BRR kway=2") 
    # # plt.plot(space,np.mean(B_kt_D,0),linestyle='dashdot',c='orange',label = "BRR kway=3") 
    # # plt.plot(space,np.mean(B_kf_D,0),linestyle='dashdot',c='green',label = "BRR kway=4") 
    
    # plt.ylim(0,1)
    # plt.legend()
    # plt.xlabel("Users", fontsize=16)
    # plt.ylabel("AVD", fontsize=16)
    # plt.grid(True)
    # plt.tight_layout()
    # chin='Plots/0.9.jpg' 
    # plt.savefig(chin)
    # plt.close()      
     
    
    # figure(figsize=(27, 15), dpi=100)
    # fbp = ['0.3','0.5','0.7','0.9']
    # l2 = [ np.mean(L_ktwA,0)[-1],np.mean(L_ktwB,0)[-1],np.mean(L_ktwC,0)[-1],np.mean(L_ktwD,0)[-1]]
    # l3 = [ np.mean(L_ktA,0)[-1],np.mean(L_ktB,0)[-1],np.mean(L_ktC,0)[-1],np.mean(L_ktD,0)[-1]]
    # l4 = [ np.mean(L_kfA,0)[-1],np.mean(L_kfB,0)[-1],np.mean(L_kfC,0)[-1],np.mean(L_kfD,0)[-1]]
    # v2 = [ np.mean(V_ktwA,0)[-1],np.mean(V_ktwB,0)[-1],np.mean(V_ktwC,0)[-1],np.mean(V_ktwD,0)[-1]]
    # v3 = [ np.mean(V_ktA,0)[-1],np.mean(V_ktB,0)[-1],np.mean(V_ktC,0)[-1],np.mean(V_ktD,0)[-1]]
    # v4 = [ np.mean(V_kfA,0)[-1],np.mean(V_kfB,0)[-1],np.mean(V_kfC,0)[-1],np.mean(V_kfD,0)[-1]]
    # space = [0,0,0,0]
    
    # x_axis = np.arange(len(fbp))
    
    # plt.bar(x_axis -0.3, l2, width=0.1, label = 'LASSO 2-way',fill=False, hatch='/')
    # plt.bar(x_axis +0.0, v2, width=0.1, label = 'VAE 2-way',color = 'blue', hatch='/')
    
    # plt.bar(x_axis -0.2, l3, width=0.1, label = 'LASSO 3-way',fill=False, hatch='o')
    # plt.bar(x_axis +0.1, v3, width=0.1, label = 'VAE 3-way',color = 'orange', hatch='o')
    
    # plt.bar(x_axis -0.1, l4, width=0.1, label = 'LASSO 4-way',fill=False, hatch='x')
    # plt.bar(x_axis +0.2, v4, width=0.1, label = 'VAE 4-way',color = 'green', hatch='x')
    
    # plt.xticks(x_axis,fbp)
    # plt.xticks(fontsize=22)
    # plt.yticks(fontsize=22)
    # plt.legend(loc='best')
    # plt.legend(fontsize=22)
    # plt.ylim(0,1)
    # plt.xlabel("Flip Bit Probability",fontsize=22)
    # plt.ylabel("AVD",fontsize=22)
    # chin='Plots/Final.jpg' 
    # plt.savefig(chin)
    # plt.close()    
    
    
L_ktw_A=np.array(L_ktw_A)
L_ktw_B=np.array(L_ktw_B)
L_ktw_C=np.array(L_ktw_C)
L_ktw_D=np.array(L_ktw_D)

L_kt_A=np.array(L_kt_A)
L_kt_B=np.array(L_kt_B)
L_kt_C=np.array(L_kt_C)
L_kt_D=np.array(L_kt_D)

L_kf_A=np.array(L_kf_A)
L_kf_B=np.array(L_kf_B)
L_kf_C=np.array(L_kf_C)
L_kf_D=np.array(L_kf_D)

V_ktw_A=np.array(V_ktw_A)
V_ktw_B=np.array(V_ktw_B)
V_ktw_C=np.array(V_ktw_C)
V_ktw_D=np.array(V_ktw_D)

V_kt_A=np.array(V_kt_A)
V_kt_B=np.array(V_kt_B)
V_kt_C=np.array(V_kt_C)
V_kt_D=np.array(V_kt_D)

V_kf_A=np.array(V_kf_A)
V_kf_B=np.array(V_kf_B)
V_kf_C=np.array(V_kf_C)
V_kf_D=np.array(V_kf_D)


B_ktw_A=np.array(B_ktw_A)
B_ktw_B=np.array(B_ktw_B)
B_ktw_C=np.array(B_ktw_C)
B_ktw_D=np.array(B_ktw_D)

B_kt_A=np.array(B_kt_A)
B_kt_B=np.array(B_kt_B)
B_kt_C=np.array(B_kt_C)
B_kt_D=np.array(B_kt_D)

B_kf_A=np.array(B_kf_A)
B_kf_B=np.array(B_kf_B)
B_kf_C=np.array(B_kf_C)
B_kf_D=np.array(B_kf_D)

######################



####0.1
dL = []
dV = []
sL = []
sV = []
dfTT = []
dL =np.array([np.mean(L_ktw_A,0)[-1],np.mean(L_kt_A,0)[-1],np.mean(L_kf_A,0)[-1]])
dV =np.array([np.mean(V_ktw_A,0)[-1],np.mean(V_kt_A,0)[-1],np.mean(V_kf_A,0)[-1]])
sL =np.array([np.std(L_ktw_A,0)[-1],np.std(L_kt_A,0)[-1],np.std(L_kf_A,0)[-1]])
sV =np.array([np.std(V_ktw_A,0)[-1],np.std(V_kt_A,0)[-1],np.std(V_kf_A,0)[-1]])
Ecolumns = ["2", "3",  "4"]
Erows = ["Lasso",  "Std Lasso", "VAE", "Std VAE"]
dfTT = []
dfTT = pd.DataFrame(data=np.array([dL,sL,dV,sV]), index=Erows, columns=Ecolumns)
dfTT.to_csv('0.3_.csv', index=True)

####0.4
dL = []
dV = []
sL = []
sV = []
dfTT = []
dL =np.array([np.mean(L_ktw_B,0)[-1],np.mean(L_kt_B,0)[-1],np.mean(L_kf_B,0)[-1]])
dV =np.array([np.mean(V_ktw_B,0)[-1],np.mean(V_kt_B,0)[-1],np.mean(V_kf_B,0)[-1]])
sL =np.array([np.std(L_ktw_B,0)[-1],np.std(L_kt_B,0)[-1],np.std(L_kf_B,0)[-1]])
sV =np.array([np.std(V_ktw_B,0)[-1],np.std(V_kt_B,0)[-1],np.std(V_kf_B,0)[-1]])
Ecolumns = ["2", "3",  "4"]
Erows = ["Lasso",  "Std Lasso", "VAE", "Std VAE"]
dfTT = []
dfTT = pd.DataFrame(data=np.array([dL,sL,dV,sV]), index=Erows, columns=Ecolumns)
dfTT.to_csv('0.5_.csv', index=True)
####0.7
dL = []
dV = []
sL = []
sV = []
dfTT = []
dL =np.array([np.mean(L_ktw_C,0)[-1],np.mean(L_kt_C,0)[-1],np.mean(L_kf_C,0)[-1]])
dV =np.array([np.mean(V_ktw_C,0)[-1],np.mean(V_kt_C,0)[-1],np.mean(V_kf_C,0)[-1]])
sL =np.array([np.std(L_ktw_C,0)[-1],np.std(L_kt_C,0)[-1],np.std(L_kf_C,0)[-1]])
sV =np.array([np.std(V_ktw_C,0)[-1],np.std(V_kt_C,0)[-1],np.std(V_kf_C,0)[-1]])
Ecolumns = ["2", "3",  "4"]
Erows = ["Lasso",  "Std Lasso", "VAE", "Std VAE"]
dfTT = []
dfTT = pd.DataFrame(data=np.array([dL,sL,dV,sV]), index=Erows, columns=Ecolumns)
dfTT.to_csv('0.7_.csv', index=True)
####0.9
dL = []
dV = []
sL = []
sV = []
dfTT = []
dL =np.array([np.mean(L_ktw_D,0)[-1],np.mean(L_kt_D,0)[-1],np.mean(L_kf_D,0)[-1]])
dV =np.array([np.mean(V_ktw_D,0)[-1],np.mean(V_kt_D,0)[-1],np.mean(V_kf_D,0)[-1]])
sL =np.array([np.std(L_ktw_D,0)[-1],np.std(L_kt_D,0)[-1],np.std(L_kf_D,0)[-1]])
sV =np.array([np.std(V_ktw_D,0)[-1],np.std(V_kt_D,0)[-1],np.std(V_kf_D,0)[-1]])
Ecolumns = ["2", "3",  "4"]
Erows = ["Lasso",  "Std Lasso", "VAE", "Std VAE"]
dfTT = []
dfTT = pd.DataFrame(data=np.array([dL,sL,dV,sV]), index=Erows, columns=Ecolumns)
dfTT.to_csv('0.9_.csv', index=True)




plt.plot(space,np.mean(L_ktw_A,0),linestyle='dashed',c='blue',label = "LASSO-kway=2") 
plt.plot(space,np.mean(L_kt_A,0),linestyle='dashed',c='orange',label = "LASSO-kway=3") 
plt.plot(space,np.mean(L_kf_A,0),linestyle='dashed',c='green',label = "LASSO-kway=4") 

plt.plot(space,np.mean(V_ktw_A,0),c='blue',label = "VAE kway=2") 
plt.plot(space,np.mean(V_kt_A,0),c='orange',label = "VAE kway=3") 
plt.plot(space,np.mean(V_kf_A,0),c='green',label = "VAE kway=4") 

# plt.plot(space,np.mean(B_ktw_A,0),linestyle='dashdot',c='blue',label = "BRR kway=2") 
# plt.plot(space,np.mean(B_kt_A,0),linestyle='dashdot',c='orange',label = "BRR kway=3") 
# plt.plot(space,np.mean(B_kf_A,0),linestyle='dashdot',c='green',label = "BRR kway=4") 

plt.ylim(0,1)
plt.legend()
plt.xlabel("Users", fontsize=16)
plt.ylabel("AVD", fontsize=16)
plt.grid(True)
plt.tight_layout()
chin='Plots/0.3.jpg' 
plt.savefig(chin)
plt.close()     
###########
plt.plot(space,np.mean(L_ktw_B,0),linestyle='dashed',c='blue',label = "LASSO-kway=2") 
plt.plot(space,np.mean(L_kt_B,0),linestyle='dashed',c='orange',label = "LASSO-kway=3") 
plt.plot(space,np.mean(L_kf_B,0),linestyle='dashed',c='green',label = "LASSO-kway=4") 

plt.plot(space,np.mean(V_ktw_B,0),c='blue',label = "VAE kway=2") 
plt.plot(space,np.mean(V_kt_B,0),c='orange',label = "VAE kway=3") 
plt.plot(space,np.mean(V_kf_B,0),c='green',label = "VAE kway=4") 

# plt.plot(space,np.mean(B_ktw_B,0),linestyle='dashdot',c='blue',label = "BRR kway=2") 
# plt.plot(space,np.mean(B_kt_B,0),linestyle='dashdot',c='orange',label = "BRR kway=3") 
# plt.plot(space,np.mean(B_kf_B,0),linestyle='dashdot',c='green',label = "BRR kway=4") 
plt.ylim(0,1)
plt.legend()
plt.xlabel("Users", fontsize=16)
plt.ylabel("AVD", fontsize=16)
plt.grid(True)
plt.tight_layout()
chin='Plots/0.5.jpg' 
plt.savefig(chin)
plt.close()     
#########
plt.plot(space,np.mean(L_ktw_C,0),linestyle='dashed',c='blue',label = "LASSO-kway=2") 
plt.plot(space,np.mean(L_kt_C,0),linestyle='dashed',c='orange',label = "LASSO-kway=3") 
plt.plot(space,np.mean(L_kf_C,0),linestyle='dashed',c='green',label = "LASSO-kway=4") 

plt.plot(space,np.mean(V_ktw_C,0),c='blue',label = "VAE kway=2") 
plt.plot(space,np.mean(V_kt_C,0),c='orange',label = "VAE kway=3") 
plt.plot(space,np.mean(V_kf_C,0),c='green',label = "VAE kway=4")

# plt.plot(space,np.mean(B_ktw_C,0),linestyle='dashdot',c='blue',label = "BRR kway=2") 
# plt.plot(space,np.mean(B_kt_C,0),linestyle='dashdot',c='orange',label = "BRR kway=3") 
# plt.plot(space,np.mean(B_kf_C,0),linestyle='dashdot',c='green',label = "BRR kway=4") 
plt.ylim(0,1)
plt.legend()
plt.xlabel("Users", fontsize=16)
plt.ylabel("AVD", fontsize=16)
plt.grid(True)
plt.tight_layout()
chin='Plots/0.7.jpg' 
plt.savefig(chin)
plt.close()     
#########
plt.plot(space,np.mean(L_ktw_D,0),linestyle='dashed',c='blue',label = "LASSO-kway=2") 
plt.plot(space,np.mean(L_kt_D,0),linestyle='dashed',c='orange',label = "LASSO-kway=3") 
plt.plot(space,np.mean(L_kf_D,0),linestyle='dashed',c='green',label = "LASSO-kway=4") 

plt.plot(space,np.mean(V_ktw_D,0),c='blue',label = "VAE kway=2") 
plt.plot(space,np.mean(V_kt_D,0),c='orange',label = "VAE kway=3") 
plt.plot(space,np.mean(V_kf_D,0),c='green',label = "VAE kway=4") 

# plt.plot(space,np.mean(B_ktw_D,0),linestyle='dashdot',c='blue',label = "BRR kway=2") 
# plt.plot(space,np.mean(B_kt_D,0),linestyle='dashdot',c='orange',label = "BRR kway=3") 
# plt.plot(space,np.mean(B_kf_D,0),linestyle='dashdot',c='green',label = "BRR kway=4") 

plt.ylim(0,1)
plt.legend()
plt.xlabel("Users", fontsize=16)
plt.ylabel("AVD", fontsize=16)
plt.grid(True)
plt.tight_layout()
chin='Plots/0.9.jpg' 
plt.savefig(chin)
plt.close()      
 

figure(figsize=(27, 15), dpi=100)
fbp = ['0.3','0.5','0.7','0.9']
l2 = [ np.mean(L_ktw_A,0)[-1],np.mean(L_ktw_B,0)[-1],np.mean(L_ktw_C,0)[-1],np.mean(L_ktw_D,0)[-1]]
l3 = [ np.mean(L_kt_A,0)[-1],np.mean(L_kt_B,0)[-1],np.mean(L_kt_C,0)[-1],np.mean(L_kt_D,0)[-1]]
l4 = [ np.mean(L_kf_A,0)[-1],np.mean(L_kf_B,0)[-1],np.mean(L_kf_C,0)[-1],np.mean(L_kf_D,0)[-1]]
v2 = [ np.mean(V_ktw_A,0)[-1],np.mean(V_ktw_B,0)[-1],np.mean(V_ktw_C,0)[-1],np.mean(V_ktw_D,0)[-1]]
v3 = [ np.mean(V_kt_A,0)[-1],np.mean(V_kt_B,0)[-1],np.mean(V_kt_C,0)[-1],np.mean(V_kt_D,0)[-1]]
v4 = [ np.mean(V_kf_A,0)[-1],np.mean(V_kf_B,0)[-1],np.mean(V_kf_C,0)[-1],np.mean(V_kf_D,0)[-1]]
space = [0,0,0,0]

x_axis = np.arange(len(fbp))

plt.bar(x_axis -0.3, l2, width=0.1, label = 'LASSO 2-way',fill=False, hatch='/')
plt.bar(x_axis +0.0, v2, width=0.1, label = 'VAE 2-way',color = 'blue', hatch='/')

plt.bar(x_axis -0.2, l3, width=0.1, label = 'LASSO 3-way',fill=False, hatch='o')
plt.bar(x_axis +0.1, v3, width=0.1, label = 'VAE 3-way',color = 'orange', hatch='o')

plt.bar(x_axis -0.1, l4, width=0.1, label = 'LASSO 4-way',fill=False, hatch='x')
plt.bar(x_axis +0.2, v4, width=0.1, label = 'VAE 4-way',color = 'green', hatch='x')

plt.xticks(x_axis,fbp)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(loc='best')
plt.legend(fontsize=22)
plt.ylim(0,1)
plt.xlabel("Flip Bit Probability",fontsize=22)
plt.ylabel("AVD",fontsize=22)
chin='Plots/Final.jpg' 
plt.savefig(chin)
plt.close()      
 


