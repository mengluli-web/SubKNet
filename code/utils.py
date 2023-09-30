# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 19:41:08 2023

@author: lml
"""

import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef,roc_curve
import pandas as pd
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')
seed=1
torch.manual_seed(seed)
np.random.seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
def get_aupr(pre,rec):
    pr_value=0.0
    for ii in range(len(rec[:-1])):
        x_r,x_l=rec[ii],rec[ii+1]
        y_t,y_b=pre[ii],pre[ii+1]
        tempo=abs(x_r-x_l)*(y_t+y_b)*0.5
        pr_value+=tempo
    return pr_value

def comput_scores(y_test, y_pred, th=0.5):           
    y_predlabel = [(0. if item < th else 1.) for item in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SPE = tn*1./(tn+fp)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    fpr,tpr,threshold = roc_curve(y_test, y_predlabel)
    sen, spe, pre, f1, mcc, acc, auc, tn, fp, fn, tp = np.array([recall_score(y_test, y_predlabel), SPE, precision_score(y_test, y_predlabel), 
                                                                  f1_score(y_test, y_predlabel), MCC, accuracy_score(y_test, y_predlabel), 
                                                                  roc_auc_score(y_test, y_pred), tn, fp, fn, tp])
    precision,recall,_ =precision_recall_curve(y_test, y_pred)
    aupr=get_aupr(precision,recall)
    return [aupr, auc, f1, acc, sen, spe, pre]

def comput_metrics(outputs,targets):
    score1=[]
    for i in np.array(outputs):
        if sum(i)!=0:
            score1.append((i/sum(i)).tolist())
        else:
            score1.append([0,0])
    score2=np.array(score1)[:,1].tolist()
    cvs=comput_scores(targets,score2)  
    return cvs

def one_hot(labels, max_n):
    labels = torch.LongTensor(labels)
    out_tensor = torch.nn.functional.one_hot(labels, int(max_n)+1)
    return sp.coo_matrix(out_tensor.type(torch.FloatTensor).detach().numpy())
    
def split_datas(dataset, kfs, seed,types):
    np.random.seed(seed)
    kf = KFold(n_splits=kfs, shuffle=True)
    kfolds = kf.get_n_splits()
    
    if dataset=='PPI':
        ###this dataset include positive and negative samples
        protein=pd.read_csv('../data/'+dataset+'/protein_name.txt',header=None).values.tolist()
        dic_pro={}
        for i in range(len(protein)):
            dic_pro[protein[i][0]]=i
        data1=pd.read_csv('../data/'+dataset+'/protein.actions.tsv',header=None,sep='\t')
        idx=[]
        idx_zero=[]
        for i in data1.index:
            if data1.loc[i,0]==data1.loc[i,1]:
                continue
            if data1.loc[i,2]==1:
                idx.append([dic_pro[data1.loc[i,0]],dic_pro[data1.loc[i,1]],data1.loc[i,2]])
            else:
                idx_zero.append([dic_pro[data1.loc[i,0]],dic_pro[data1.loc[i,1]],data1.loc[i,2]])
        idx_new=[]
        for i in range(len(idx)):
            if idx[i][1]>idx[i][0]:
                idx_new.append([idx[i][1],idx[i][0],idx[i][2]])
            else:
                idx_new.append([idx[i][0],idx[i][1],idx[i][2]])
        idx=idx_new
        idx_zero_new=[]
        for i in range(len(idx_zero)):
            if idx_zero[i][1]>idx_zero[i][0]:
                idx_zero_new.append([idx_zero[i][1],idx_zero[i][0],idx_zero[i][2]])
            else:
                idx_zero_new.append([idx_zero[i][0],idx_zero[i][1],idx_zero[i][2]])
        idx_zero=idx_zero_new       
        np.random.shuffle(idx)
        np.random.shuffle(idx_zero)
        assos=[]
        for i in range(len(idx)):
            assos.append((idx[i][0],idx[i][1],1))
        idx_train_val_pos=assos[:int(len(assos)*9/10)]
        idx_test_pos=assos[int(len(assos)*9/10):]

        assos=[]
        for i in range(len(idx_zero)):
            assos.append((idx_zero[i][0],idx_zero[i][1],0))
        
        idx_train_val_neg=assos[:int(len(assos)*9/10)]
        idx_test_neg=assos[int(len(assos)*9/10):]
        idx_test=idx_test_pos+idx_test_neg
        np.random.shuffle(idx_test)
        idx_train_val=idx_train_val_pos+idx_train_val_neg
        np.random.shuffle(idx_train_val)
        kfs = [train_test for train_test in kf.split(idx_train_val)]
        data1=np.zeros((len(protein),len(protein)))
        nums=0
        for i in idx:
            if i[2]==1:
                nums+=1
                data1[i[0],i[1]]=1
                data1[i[1],i[0]]=1
    else:
        ###in addition to PPI dataset, other datasets need to select negative samples
        data1=np.loadtxt('../data/'+dataset+'/inters.txt',delimiter='\t')
        data1[data1==-1]=1
        idx=np.where(data1 != 0)
        idx=np.concatenate((idx[0].reshape((-1,1)),idx[1].reshape((-1,1))),axis=1).tolist()   
        if types=='homo':
            idx_new=[]
            for i in range(len(idx)):
                if idx[i][1]>=idx[i][0]:
                    continue
                else:
                    idx_new.append(idx[i])
            idx=idx_new
        np.random.shuffle(idx)
        idx_zero=np.where(data1 == 0)
        idx_zero=np.concatenate((idx_zero[0].reshape((-1,1)),idx_zero[1].reshape((-1,1))),axis=1).tolist()
        if types=='homo':
            idx_zero_new=[]
            for i in range(len(idx_zero)):
                if idx_zero[i][1]>=idx_zero[i][0]:
                    continue
                else:
                    idx_zero_new.append(idx_zero[i])
            idx_zero=idx_zero_new 
        idx_zero2=[]   ##remove self-loops
        for i in idx_zero:
            if i[0]!=i[1]:
               idx_zero2.append(i) 
        np.random.shuffle(idx_zero2)
        other_neg=idx_zero2[len(idx):]
        idx_zero2=idx_zero2[:len(idx)]
        assos=[]
        for i in range(len(idx)):
            assos.append((idx[i][0],idx[i][1],1))
        
        idx_train_val_pos=assos[:int(len(assos)*9/10)]
        idx_test_pos=assos[int(len(assos)*9/10):]
    
        assos=[]
        for i in range(len(idx_zero2)):
            assos.append((idx_zero2[i][0],idx_zero2[i][1],0))
        assos_other=[]
        for i in range(len(other_neg)):
            assos_other.append((other_neg[i][0],other_neg[i][1],0))
    
        idx_train_val_neg=assos[:int(len(assos)*9/10)]+assos_other
        idx_test_neg=assos[int(len(assos)*9/10):]
        idx_test=idx_test_pos+idx_test_neg
        np.random.shuffle(idx_test)
        np.random.seed(seed)
        # print(seed)
        np.random.shuffle(idx_train_val_neg)
        idx_train_val_neg=idx_train_val_neg[:int(len(assos)*9/10)]
        idx_train_val=idx_train_val_pos+idx_train_val_neg
        np.random.shuffle(idx_train_val)
        kfs = [train_test for train_test in kf.split(idx_train_val)]
        if types=='homo':
            data1=np.zeros((data1.shape))
            for i in idx_train_val+idx_test:
                if i[2]==1:
                    data1[i[0],i[1]]=1
                    data1[i[1],i[0]]=1
        
    np.savez('../data/split_data_'+dataset+'.npz',
              md=data1,
              idx_train_val=idx_train_val,
              idx_test=idx_test, 
              kfolds=kfolds,
              kfs=kfs)

def loadsims(dataset):
    sims_m=np.loadtxt('../data/'+dataset+'/sims_m.txt',delimiter='\t')
    sims_d=np.loadtxt('../data/'+dataset+'/sims_d.txt',delimiter='\t')
    ###avoid error when the node number is small than 128
    if min(sims_m.shape[0],sims_d.shape[0])>=128:
        pca = PCA(n_components=128)
    else:
        pca = PCA(n_components=min(sims_m.shape[0],sims_d.shape[0]))
    if dataset=='LuoDTI':
        sims_m1=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_protein.txt',delimiter='\t'))
        sims_m2=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_protein_protein.txt',delimiter='\t'))
        sims_m3=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_protein_disease.txt',delimiter='\t'))
        sims_d1=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_drug.txt',delimiter='\t'))
        sims_d2=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_drug_drug.txt',delimiter='\t'))
        sims_d3=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_drug_disease.txt',delimiter='\t'))
        sims_d4=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_drug_se.txt',delimiter='\t'))
        sims_m=np.average([sims_m1,sims_m2,sims_m3],axis=0)
        sims_d=np.average([sims_d1,sims_d2,sims_d3,sims_d4],axis=0)
    elif dataset=='ZhangDDA':
        sims_m1=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_m.txt',delimiter='\t'))
        sims_m2=pca.fit_transform(np.loadtxt('../data/'+dataset+'/enzyme_sim.txt',delimiter='\t'))
        sims_m3=pca.fit_transform(np.loadtxt('../data/'+dataset+'/target_sim.txt',delimiter='\t'))
        sims_m4=pca.fit_transform(np.loadtxt('../data/'+dataset+'/structure_sim.txt',delimiter='\t'))
        sims_m5=pca.fit_transform(np.loadtxt('../data/'+dataset+'/pathway_sim.txt',delimiter='\t'))

        sims_d1=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_d.txt',delimiter='\t'))
        sims_m=np.average([sims_m1,sims_m2,sims_m3,sims_m4,sims_m5],axis=0)
        sims_d=sims_d1
    elif dataset=='PPI':
        sims_m1=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_m.txt',delimiter='\t'))
        sims_m=sims_m1
        sims_d=sims_m1
    else:
        sims_m1=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_m.txt',delimiter='\t'))
        sims_d1=pca.fit_transform(np.loadtxt('../data/'+dataset+'/sims_d.txt',delimiter='\t'))
        sims_m=sims_m1
        sims_d=sims_d1
    return sims_m,sims_d

def comp_feat(MD,dataset):
    labels_onehot_M=[]
    for i in range(MD.shape[0]):
        labels_onehot_M.append(i)
    labels_onehot_D=[]
    for i in range(MD.shape[1]):
        labels_onehot_D.append(i+MD.shape[0])
    Ms1 = one_hot(labels_onehot_M, MD.shape[0]+MD.shape[1]).toarray()
    Ds1 = one_hot(labels_onehot_D, MD.shape[0]+MD.shape[1]).toarray()
    Ms3,Ds3=loadsims(dataset)
    return Ms1,Ds1,Ms3,Ds3

def comp_feat2(MD,dataset):
    labels_onehot_M=[]
    for i in range(MD.shape[0]):
        labels_onehot_M.append(i)
    Ms1 = one_hot(labels_onehot_M, MD.shape[0]).toarray()
    Ms3,Ds3=loadsims(dataset)
    return Ms1,Ms1,Ms3,Ms3
