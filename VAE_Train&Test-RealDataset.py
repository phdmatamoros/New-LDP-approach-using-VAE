# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import init
import argparse
import os
from sklearn.model_selection import train_test_split
import glob
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from torchsummary import summary
import itertools
import pandas as pd
import warnings
import gc
import random
import secrets
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_revgrad import RevGrad
from sklearn import metrics

def mmd(X, Y, gamma=1.0):
    """MMD using gaussian kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


warnings.filterwarnings("ignore")
filetxt = open('Results.txt', "a+")
filetxt.write("Epoch Patience latent Space FlipBIT Attribute Min_loss"+'\n')
filetxt.close()
##########################################
data_lab=pd.read_csv('label_server.csv') 
data_hash=pd.read_csv('hash_server.csv')
del data_lab[data_lab.columns[0]]
del data_hash[data_hash.columns[0]]
#########################################
unique=[]
for col in data_lab.columns:
    unique.append((data_lab[col]).unique())

names_col=[]
for col in data_lab.columns:
    names_col.append(col)

# for i in range(len(data_lab.columns)):
#     print(unique[i])

index=[]
c=0
for col in data_lab.columns:
    aux=[]
    for label in unique[c]:
        aux.append(data_lab.index[data_lab[col] == label].tolist())
    index.append(aux)
    c=c+1
########################################
c=0
code=[]
for col in data_hash.columns:
    b=0
    caux=[]
    for label in unique[c]:
        string=[]
        string=data_hash[col][index[c][b][0]]
        b=b+1
        aux=[]
        for s in string:
            if(s=='0') or (s=='1'):
                aux.append(float(s))
        caux.append(aux)
    code.append(caux)
    c=c+1
del aux,b,c,caux,col,label,s,string
torch.cuda.empty_cache()
gc.collect()
print('4D latent space')
LS='4D'

for dima in range(len(code)):
    names=[]
    names=names_col[dima]
    print("##################################################################")
    print("Atrribute name",names)
    print(unique[dima])
    #####################################
    ##########################
    checkdat =[1000]
    fp=[0.3,0.5,0.7,0.9]
    checklabel=[]
    
    probexp=np.zeros(len(checkdat))
    min_error=np.zeros(len(checkdat))
    
    for fptra in fp:    
        print("-------------------------------------------------------------------")
        for mask_ind in range(0,len(checkdat)):
            print("///////////////////////////////////////////////////////////////////")
            print("Training by element",str(checkdat[mask_ind]))
            print("Flip bit probability",fptra)
            times=checkdat[mask_ind]
            cadena=str(checkdat[mask_ind])
            fptracad='fptra_'+str(fptra)
            checklabel.append(cadena)
            min_error[mask_ind]
            torch.cuda.empty_cache()
            gc.collect()
            #######################################################################
            #######################################################################
            classes=[]
            binary=[]
            classes=unique[dima]
            binary=code[dima]
            CL=len(unique[dima])
            FL=len(code[dima][0])
            print("Classes",CL,', string size',FL)
           
            
            aux=[]

            for r in itertools.product(binary):
                aux.append(np.array((r[0]),dtype='float64'))    
                 
            Labels=[]
            for element in aux:
                 Labels.append(np.ones(FL)*element)

            Cboth=[]
            for element in aux:
                 Cboth.append(np.ones(( times, FL))*element)
             
            Cboth=np.array(Cboth)
            Cboth=Cboth.reshape(CL*times,FL)
            len_string = len(code[dima][0])
                          
            print('Train dataset',CL*times)
 
            ########################################################################
            ########################################################################
             
             
            perturb=[]#Cboth
            [ha,bia]=(Cboth.shape)
        
            def fun_aghm(bit,fbp):
                 #secretsGen=secrets.SystemRandom()
                 p_sample=random.uniform(0, 1)#secretsGen.randint(0,100000)/100000
                 sample=bit
                 if p_sample < fbp:############################################################################################CHANGEEEE
                     sample=random.choice([0,1])
                 return sample
             
                
            # def fun_aghm(bit,fptra):
            #      secretsGen=secrets.SystemRandom()
            #      p_sample=secretsGen.randint(0,100000)/100000
            #      sample=bit
            #      if p_sample <= fptra:############################################################################################CHANGEEEE
            #          sample=random.choice([0,1])
            #      return sample
             
            for rows in range(ha):
                 vec=[]
                 vec=Cboth[rows][:]
                 new=[fun_aghm(bit,fptra) for bit in vec]
                 perturb.append(new)
            
            perturb=np.array(perturb)
            x_train=[]
            X=perturb
            Y=Cboth
            del perturb,Cboth
            
            
            
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
            
            n_train = len(x_train)
            n_test = len(x_test)
            
            device=torch.device('cuda:0')
            
            batch_size = 16#32
            
            x_train=torch.Tensor(x_train)
            y_train=torch.Tensor(y_train)
            x_test=torch.Tensor(x_test)
            y_test=torch.Tensor(y_test)
            
            
            # print(x_train.shape)
            
            train_ds = TensorDataset(x_train, y_train)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            test_ds = TensorDataset(x_test, y_test)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            del X,Y
            
       
            #######################################################################
            class Encoder(nn.Module):
                def __init__(self, z_dim,input_dim):
                    super(Encoder, self).__init__()
                    self.conv1 = nn.Conv1d(1, 16, 2)
                    self.conv2 = nn.Conv1d(16, 16, 2)
                    self.conv3 = nn.Conv1d(16, 32, 2)
                    self.conv4 = nn.Conv1d(32, 32, 2)
                    self.fc1 = nn.Linear(32 * (input_dim-4), 64)
                    self.fc2 = nn.Linear(64, 16)
                    self.fc21 = nn.Linear(16, z_dim)
                    self.fc22 = nn.Linear(16, z_dim)
                    self.bn1 = nn.BatchNorm1d(16)
                    self.bn2 = nn.BatchNorm1d(16)
                    self.bn3 = nn.BatchNorm1d(32)
                    self.bn4 = nn.BatchNorm1d(32)
                    self.bn5 = nn.BatchNorm1d(64)
                    self.relu = nn.ReLU()
            
                def forward(self, x,input_dim):
                    # print(x.shape)
                    x = self.relu(self.conv1(x))
                    x = self.bn1(x)
                    # print(x.shape)
                    x = F.dropout(x, 0.3)
                    x = self.relu(self.conv2(x))
                    x = self.bn2(x)
                    # print(x.shape)
                    x = F.dropout(x, 0.3)
                    x = self.relu(self.conv3(x))
                    x = self.bn3(x)
                    # print(x.shape)
                    x = F.dropout(x, 0.3)
                    x = self.relu(self.conv4(x))
                    x = self.bn4(x)
                    # print(x.shape)
                    x = F.dropout(x, 0.3)
                    x = x.contiguous().view(-1, 32 * (input_dim-4))#-12
                    # print("a",x.shape)
                    x = self.relu(self.fc1(x))
                    # print("a",x.shape)
                    x = self.bn5(x)
                    x = self.relu(self.fc2(x))
                    # print(x.shape)
                    z_loc = self.fc21(x)
                    z_scale = self.fc22(x)
    
                    return z_loc, z_scale
            
            
            class Decoder(nn.Module):
                def __init__(self, z_dim,input_dim):
                    super(Decoder, self).__init__()
                    self.fc1 = nn.Linear(z_dim, 32 * (input_dim-4))
                    self.conv1 = nn.ConvTranspose1d(32, 32, 2)
                    #self.conv2 = nn.ConvTranspose1d(32, 32, 2)
                    self.conv3 = nn.ConvTranspose1d(32, 16, 2)
                    self.conv4 = nn.ConvTranspose1d(16, 16, 2)
                    self.conv5 = nn.ConvTranspose1d(16, 1, 2)
                    self.bn1 = nn.BatchNorm1d(32)
                    self.bn2 = nn.BatchNorm1d(32)
                    self.bn3 = nn.BatchNorm1d(16)
                    self.bn4 = nn.BatchNorm1d(16)
                    self.relu = nn.ReLU()
            
                def forward(self, z,input_dim):
                    # print(z.shape)
                    z = self.relu(self.fc1(z))
                    z = z.view(-1, 32,(input_dim-4))
                    # print(z.shape)
                    z = self.relu(self.conv1(z))
                    z = self.bn1(z)
                    # print(z.shape)
                    # z = self.relu(self.conv2(z))
                    # z = self.bn2(z)
                    # print(z.shape)
                    z = self.relu(self.conv3(z))
                    z = self.bn3(z)
                    # print(z.shape)
                    z = self.relu(self.conv4(z))
                    z = self.bn4(z)
                    # print(z.shape)
                    z = self.conv5(z)
                    # print(z.shape)
                    recon = torch.sigmoid(z)
                    # print(recon)
                    return recon
         
            
            
            class VAE(nn.Module):
                def __init__(self, z_dim=4,input_dim=len_string):
                    super(VAE, self).__init__()
                    self.encoder = Encoder(z_dim,input_dim)
                    self.decoder = Decoder(z_dim,input_dim)
                    self.cuda()
                    self.z_dim = z_dim
            
                def reparameterize(self, z_loc, z_scale):
                    std = z_scale.mul(0.5).exp_()
                    epsilon = torch.randn(*z_loc.size()).to(device)
                    z = z_loc + std * epsilon
                    return z

            device = torch.device("cuda:0")
            #######################################################################
            vae = VAE()
            np.zeros(len(checkdat))

            
            optimizer = torch.optim.Adam(vae.parameters(), lr=0.0001)
            #optimizer = torch.optim.RMSprop(vae.parameters(), lr=0.001, alpha=0.9)
            
            def loss_fn(recon_x, x, z_loc, z_scale):
                BCE = F.mse_loss(recon_x, x, size_average=False)*100
                KLD = -0.5 * torch.sum(1 + z_scale - z_loc.pow(2) - z_scale.exp())
                return BCE + KLD
            

        
            

            print("-----------Training------------")
            for epoch in range(500):#1000
                for x, y in train_dl:
                    x = x.cuda()
                    y = y.cuda()
                    #print(x.shape)
                    x = torch.unsqueeze(x, dim=1)
                    y = torch.unsqueeze(y, dim=1)
                    #print(x.shape)
                    z_loc, z_scale = vae.encoder(x,len_string )
                    z = vae.reparameterize(z_loc, z_scale)
                    aka = z.cpu().detach().numpy()
                    rd,cd = aka.shape
                    ##################### ###############
                    q_hat=np.random.normal(size=[rd,cd])
                    recon = vae.decoder(z,len_string)
                    loss = loss_fn(recon, y, z_loc, z_scale)+mmd(z.cpu().detach().numpy(), q_hat)
                    q_hat=np.random.normal(size=[batch_size,4])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                vae.eval()
                
                with torch.no_grad():
                    for i, (x, y) in enumerate(test_dl):
                        x = x.cuda()
                        y = y.cuda()
                        x = torch.unsqueeze(x, dim=1)
                        y = torch.unsqueeze(y, dim=1)
                        z_loc, z_scale = vae.encoder(x,len_string)
                        z = vae.reparameterize(z_loc, z_scale)
                        
                        aka = z.cpu().detach().numpy()
                        rd,cd = aka.shape
                        q_hat=np.random.normal(size=[rd,cd])
                        
                        
                        recon = vae.decoder(z,len_string)
                        test_loss = loss_fn(recon, y, z_loc, z_scale)+mmd(z.cpu().detach().numpy(), q_hat)

                        
                        
                        
                normalizer_test = len(test_dl.dataset)
                total_epoch_loss_test = test_loss / normalizer_test
                if epoch == 0:
                    loss_test_history = total_epoch_loss_test.item()
                    patience = 0
                else:
                    loss_test_history = np.append(loss_test_history, total_epoch_loss_test.item())

                if total_epoch_loss_test.item() < 0.000001+np.min(loss_test_history):
                    patience = 0
                    min_error[mask_ind]=np.min(loss_test_history)
                    torch.save(vae.state_dict(), 'results/_'+names+'_'+fptracad+'_'+cadena+"_VAE_model.pt")
                else:
                    patience +=1
                print(epoch, patience, total_epoch_loss_test.item(), np.min(loss_test_history))
                if patience == 32: #32
                    break
                
            
            
            print("Epoch", epoch)
            print("Patience", patience)
            print("Min loss", np.min(loss_test_history))
            
            filetxt = open('Results.txt', "a+")
            filetxt.write(" "+str(epoch)+" "+str(patience)+" "+str(LS)+" "+str(fptra)+" "+str(names)+" "+str(np.min(loss_test_history))+'\n')##########
            filetxt.close()
            
            
            
            
            # ######################################################################
            # ######################################################################
            aux=[]
            for r in itertools.product(classes):
                aux.append(r[0])
            # ######################################################################## 
            # ####################################################################### 

            
            
            
    
            
            
            # #################################
            torch.cuda.empty_cache()
            gc.collect()        
            
            x_train =x_train.cuda()
            x_train = torch.unsqueeze(x_train, dim=1)
        
            # X_enc=[]
            vae =[]        
            vae = VAE()
            vae.load_state_dict(torch.load('results/_'+names+'_'+fptracad+'_'+cadena+"_VAE_model.pt"))
            vae.eval()
            with torch.no_grad():
                X_enc, _ = vae.encoder(x_train,len_string)
                X_enc = X_enc.cpu().detach().numpy()
            cluster=[]
            cluster=X_enc
            y_enc=[]
            y_enc = y_train.cpu().detach().numpy()
            
            
            index=[]
            l=0
            index=np.ones(len(y_enc))*-1
            for label in Labels:
                c=0
                for element in y_enc:
                    if(np.array_equal(label,element)):
                        index[c]=l
                    c+=1
                l+=1
 
            
            
            labels=aux

            ################################################################################
            ################################################################################
            torch.cuda.empty_cache()
            gc.collect()
            print("---Computing on Real dataset---")
            torch.cuda.empty_cache()
            gc.collect()
            Randomtimes=[1]#,2,3,4,5,6,7,8,9,10]
            aghmnp=[]
            aghmcen=[]
            for times_random in Randomtimes:
                torch.cuda.empty_cache()
                gc.collect()
                maxproba=[]
                vecproba=[]
                outdfa = pd.DataFrame(columns=[])
                outdf = pd.DataFrame(columns=[])
           
                bit_vae=[]
                bit_per=[]
                for indexno, row in data_hash.iterrows():
                    aux1=row[names]
                    vec=[]
                    for s in aux1:
                        if(s=='0') or (s=='1'):
                            vec.append(float(s))
            
            
                    vector=[]
                    vector=np.array(vec,dtype='float64')
                    ###############################################################
                    
                    perturb=[]      
                    # print(vector)
                    perturb=[fun_aghm(bit,fptra) for bit in vector]
                    bit_per.append(np.array(perturb,dtype='int32'))
                    # print(bit_per)
                    vector=[]
                    vector=perturb
        
                    perturb=[]
                        
                    perturb = np.array(vector)  
                    
                    del vector
                    perturb = torch.from_numpy(perturb)
                    perturb = perturb.to('cuda')
                    perturb = perturb.cuda()
                    perturb = torch.unsqueeze(perturb, dim=0)
                    perturb = torch.unsqueeze(perturb, dim=0)
                    X_vec=[]
                    perturb=perturb.float()
                    vae =[]        
                    vae = VAE()
                    vae.load_state_dict(torch.load('results/_'+names+'_'+fptracad+'_'+cadena+"_VAE_model.pt"))
                    vae.eval()
                    with torch.no_grad():
                        X_vec, o_vec = vae.encoder(perturb,len_string)
                        z = vae.reparameterize(X_vec, o_vec)
                        recon = vae.decoder(z,len_string)
                        recon=torch.round(recon)
                        recon=recon.cpu().numpy()
                        X_vec = X_vec.cpu().detach().numpy()
                
                    bit_vae.append(list(recon.squeeze()))
                    recon=[]
                    del perturb
                    probclass=[]
                    probclassc=[]
                    for i in range(CL):
                        sx=[]

                        cx=[]

                        sx=cluster[:][index==i]
                        cx=np.mean(sx,axis=0)
                        
                        sx=sx-X_vec
                        sx=sx**2
                        sx=np.sum(sx,axis=1)
                        sx=sx**(1/2)
                        prob=np.exp(-(sx))
                        a=np.argmax(prob)
                        probclass.append(prob[a])
                        #####################
                        cx=cx-X_vec
                        cx=cx**2 
                        cx=np.sum(cx,axis=1)
                        cx=cx**(1/2)
                        probc=np.exp(-(cx))
                        probclassc.append(probc)
   
                        
                    sum_probclass=(probclass/sum(probclass))*100
                    
                    
                    
                    sum_probclassc=(probclassc/sum(probclassc))*100
                    sum_probclassc=np.squeeze(sum_probclassc)
                    
         
                    aghmnp.append(sum_probclass)
                    aghmcen.append(sum_probclassc)

                    
             
                    del sx,cx,a,prob,probclass
                # bit_per=np.array(bit_per)
                # 1/0
                bit_aghm=np.array(bit_vae)
                aghmnp=np.array(aghmnp)
                aghmcen=np.array(aghmcen)
                rows, columns = aghmnp.shape
                #outdfa=[]
                #outdfa=pd.DataFrame(aghmnp, columns=[aux])
                #outdfa.to_csv("results/Votes"+LS+names+'_flopprob_'+str(fptra)+'_peratt_'+str(times)+'_'+'probabilities.csv') 
                outdfb=[]
                outdfb=pd.DataFrame(aghmcen, columns=[aux])
                outdfb.to_csv("results/_"+LS+names+'_flopprob_'+str(fptra)+'_peratt_'+str(times)+'_'+'probabilities_centroid.csv')
                bittoLASSO=[]
                bittoLASSO=pd.DataFrame(bit_per)
                bittoLASSO.to_csv("results/perturb_"+names+'_flopprob_'+str(fptra)+'_peratt_'+str(times)+'_'+'.csv')
           
                
                #outdfc=[]
                #outdfc=pd.DataFrame(bit_aghm)
                #outdfc.to_csv("results/Votes"+LS+names+'_flopprob_'+str(fptra)+'_peratt_'+str(times)+'_'+'probabilities_bin.csv')            
            del outdf,rows,columns,aghmnp,sum_probclass,len_string
            del vae,z,z_loc,z_scale,x_test,x_train,y_test,y_train,y
            del aux,aux1,batch_size,bia,binary,c,cadena,classes,cluster
            del X_vec,X_enc,x,vec,vecproba,train_dl,train_ds,total_epoch_loss_test
            del times,times_random,test_dl,test_loss,test_ds
            del device
            del element,epoch,index,label,loss, loss_test_history,maxproba,optimizer
            del recon,y_enc,outdfb#,outdfa,outdfc
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
torch.cuda.empty_cache()
gc.collect()
            
        
        
