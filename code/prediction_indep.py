# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:19:39 2022

@author: lml
"""
from utils import comput_metrics
from parameters import args_parser
import numpy as np
import torch
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
args=args_parser()
class ProgressBar():
    def __init__(self, total_num, epoch, moniter=[]):
        self.__inner_width = 20
        self.__total = total_num
        self.__moniter = dict(zip(moniter, ['' for _ in moniter]))
        self.epoch = epoch

    def update(self, i, **moniter):
        if i > self.__total:
            i = self.__total
        progress = round((i/self.__total)*self.__inner_width)
        str_progress = '\rBatch:{:02d} ['.format(self.epoch)+'='*progress+'-'*(
            self.__inner_width-progress)+'] {}/{} '.format(i, self.__total)
        ext_info = ''
        for i in moniter.keys():
            ext_info += '{}:{:.4f} '.format(i, moniter[i])
        print(str_progress+ext_info, end='', flush=True)

    def __len__(self):
        return self.__total + 1
    


def preds(MDs,adj,features,nodes,y,m_feat,d_feat,bsize,epoch,model,optimizer=None,criterion=None):
    if args.types=='homo':
        MDS=MDs
    else:
        MDS=np.vstack((np.hstack((np.zeros((MDs.shape[0],MDs.shape[0])),MDs)),
                        np.hstack((MDs.T,np.zeros((MDs.shape[1],MDs.shape[1]))))))
    outputs1=[]
    targets=[]
    N = len(y)
    index = np.random.permutation(N)
    total_iters = (N + (bsize - 1) * (optimizer is None)) // bsize
    pbar = ProgressBar(total_iters, epoch=epoch+1, moniter=['loss', 'acc'])
    pos=0
    embed_local=[]
    embed_global=[]
    sub_node=[]
    embed_sub_filter=[]
    for i in range(0, N, bsize):
        n_graphs = min(i+bsize, N) - i
        x_sub_nodes=[]
        y_batch=[]
        x_sub_adj = torch.zeros(n_graphs, max([tt.shape[0] for tt in features]), max([tt.shape[0] for tt in features]))
        x_sub_feat = torch.zeros(n_graphs, max([tt.shape[0] for tt in features]), features[0].shape[1])
        for node in range(n_graphs):
            y_batch.append(y[index[i+node]])
            x_sub_nodes.append(nodes[index[i+node]])
            
            x_sub_feat[node,:features[index[i+node]].shape[0],:features[index[i+node]].shape[1]]=torch.FloatTensor(features[index[i+node]].toarray())
            x_sub_adj[node,:adj[index[i+node]].shape[0],:adj[index[i+node]].shape[1]]=torch.FloatTensor(adj[index[i+node]].toarray())

        if len(x_sub_adj)==1:
            continue
        outputs,loss2,adj_hiddens,oot,local_embed,global_embed,outputs_srl,outputs_grl= model(torch.FloatTensor(MDS).cuda(),x_sub_adj.cuda(), x_sub_feat.cuda(), 
                                            m_feat,d_feat,np.array(x_sub_nodes),args) #,adj_hidd,feat_hidd
        if len(embed_local)==0:
            embed_local=local_embed.detach().cpu().numpy()
            embed_global=global_embed.detach().cpu().numpy()
            #embed_sub_node=ssn[0].detach().cpu().numpy()
            embed_sub_filter=oot[0].detach().cpu().numpy()
            sub_node=np.array(x_sub_nodes)
        else:
            embed_local=np.concatenate((embed_local,local_embed.detach().cpu().numpy()),axis=0)
            embed_global=np.concatenate((embed_global,global_embed.detach().cpu().numpy()),axis=0)
            #embed_sub_node=np.concatenate((embed_sub_node,ssn[0].detach().cpu().numpy()),axis=0)
            embed_sub_filter=np.concatenate((embed_sub_filter,oot[0].detach().cpu().numpy()),axis=0)
            sub_node=np.concatenate((sub_node,np.array(x_sub_nodes)),axis=0)
        
        loss1=criterion(outputs, torch.LongTensor(y_batch).cuda())
        loss3=criterion(outputs_srl, torch.LongTensor(y_batch).cuda())
        loss4=criterion(outputs_grl, torch.LongTensor(y_batch).cuda())
        loss=loss1+args.lambda1*loss2+args.lambda2*loss3+args.lambda3*loss4
        #print(loss1,loss2,loss3,loss4)
        output1=outputs.detach().cpu().numpy()
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss.data.detach().cpu().numpy()
        pbar.update(pos, loss=loss)
        pos+=1
        outputs1+=output1.tolist()
        targets+=y_batch
    cvs=comput_metrics(outputs1,targets)
    return loss,outputs1,targets,cvs,adj_hiddens,embed_local,embed_global,embed_sub_filter,sub_node