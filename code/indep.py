import os 
from parameters import args_parser
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from model_indep import Model
from subgraph_extraction import links2subgraphs
from utils import *
from prediction_indep import preds
import scipy.sparse as sp
from shutil import copyfile
import shutil
import warnings
warnings.filterwarnings('ignore')
seed=1
torch.manual_seed(seed)
np.random.seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

def indep_process():
    args = args_parser()
    if not os.path.exists(args.dataset + '_result/'):
        os.mkdir(args.dataset + '_result/')
    if not os.path.exists(args.dataset + '_result/embeddings'):
        os.mkdir(args.dataset + '_result/embeddings')
    if not os.path.exists(args.dataset + '_result/models'):
        os.mkdir(args.dataset + '_result/models')
    if not os.path.exists(args.dataset + '_result/adjs'):
        os.mkdir(args.dataset + '_result/adjs')
    if not os.path.exists(args.dataset + '_result/filter_sub'):
        os.mkdir(args.dataset + '_result/filter_sub')
        
    data = np.load('../data/split_data_'+args.dataset+'.npz', allow_pickle=True)
    data1 = data['md']
    idx_train_val = data['idx_train_val']
    idx_test = data['idx_test']
    kfolds = data['kfolds']
    kfs = data['kfs']
           
    n_classes=np.unique(data1).size
    
    test_positive_edges=np.array([[int(i[0]),int(i[1])] for i in idx_test if i[2]==1])
    test_positive_edges2=np.array([[int(i[1]),int(i[0])] for i in idx_test if i[2]==1])
    test_zero_edges=np.array([[int(i[0]),int(i[1])] for i in idx_test if i[2]==0])
    
    test_positive_edges = (test_positive_edges.T[0],  test_positive_edges.T[1])
    test_positive_edges2 = (test_positive_edges2.T[0],  test_positive_edges2.T[1])
    test_zero_edges = (test_zero_edges.T[0],  test_zero_edges.T[1])  

    train_edges=idx_train_val

    MD = np.copy(data1)
    MD[test_positive_edges] = 0
    if args.types=='homo':
        MD[test_positive_edges2] = 0
        MD2=np.tril(MD, k=0)
    else:
        MD2=MD
    train_positive_edges = np.where(MD2==1)
    train_zero_edges=np.array([[int(i[0]),int(i[1])] for i in train_edges if i[2]==0])#[:len(train_positive_edges[0]),:]
    train_zero_edges = (train_zero_edges.T[0],  train_zero_edges.T[1])
    
    print(len(train_positive_edges[0]),len(train_zero_edges[0]),len(test_positive_edges[0]),len(test_zero_edges[0]))
    
    if args.types=='homo':
        MD2=MD2+MD2.T
        featM,featD,featM_sim,featD_sim=comp_feat2(MD2,args.dataset)
    else:    
        featM,featD,featM_sim,featD_sim=comp_feat(MD2,args.dataset)
    
    train_graphs, _, val_graphs, max_n = links2subgraphs(MD2, train_positive_edges,train_zero_edges,
                                                                  None, None,
                                                                  test_positive_edges, test_zero_edges,
                                                                  h=1, max_nodes_per_hop=None,featM=featM,featD=featD)
 
    n_train = len(train_graphs)
    n_val = len(val_graphs)
    tt=[]
    for ij in range(len(train_graphs)):
        tt.append(train_graphs[ij])
    np.save(args.dataset + '_result/trains_subgraphs.npy',tt)
        
    tt=[]
    for ij in range(len(val_graphs)):
        tt.append(val_graphs[ij])
    np.save(args.dataset + '_result/test_subgraphs.npy',tt)
    
    if args.fixs==0:
        size_subgraphs=int(np.mean([ii[0].shape[0] for ii in train_graphs])) 
    else:
        size_subgraphs=args.fixs 
    size_graph_filters=[int(size_subgraphs)]

    # Sampling
    adj_train = [i[0] for i in train_graphs]    
    nodes_trains = [i[2] for i in train_graphs]
    y_train = [i[3] for i in train_graphs]
    
    adj_val = [i[0] for i in val_graphs]    
    nodes_val = [i[2] for i in val_graphs]
    y_val =  [i[3] for i in val_graphs]
    
    features_train = [i[1] for i in train_graphs]
    features_val = [i[1] for i in val_graphs]
    
    features_dim = features_train[0].shape[1]
    
    # Create model
    model = Model(featM_sim.shape[1],featD_sim.shape[1], 
                  features_dim, n_classes,args,
                    max_step = args.max_step, dropout_rate = args.dropout_rate,
                    size_graph_filter = size_graph_filters)
    print(model)
    model=model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    cv_tests=[]

    best_epoch=0
    best_score=0
    counter=0
    cv_trains=[]
    for epoch in range(args.epochs):
        model.train() 
        loss_tra,output_tra,targets_tra,cv_tra,adj_hiddens1,el1,eg1,ot1,sn1=preds(MD2,adj_train,features_train,nodes_trains,y_train,
                                                      featM_sim,featD_sim,args.batch_size,epoch,model,
                                                      optimizer=optimizer,criterion=criterion)
        print('epoch: ', epoch, loss_tra,'\n',cv_tra)
        # adj_hids1.append(adj_hiddens1)
        cv_trains.append(cv_tra)

        torch.save(model, args.dataset + '_result/models/model_'+str(epoch)+'.pth')
        
        np.save(args.dataset + '_result/filter_sub/filter_sub_'+str(epoch)+'.npy',ot1)
        np.save(args.dataset + '_result/filter_sub/sub_node_'+str(epoch)+'.npy',sn1)

        tt=[]
        tt=np.concatenate((el1,np.array(targets_tra).reshape(-1,1)),axis=1)
        np.save(args.dataset + '_result/embeddings/all_local_embeddings_tra_'+str(epoch)+'.npy',tt)

        tt=[]
        tt=np.concatenate((eg1,np.array(targets_tra).reshape(-1,1)),axis=1)
        np.save(args.dataset + '_result/embeddings/all_global_embeddings_tra_'+str(epoch)+'.npy',tt)

        tt=[]
        for ij in range(adj_hiddens1[0].size(2)):
            tt.append(sp.coo_matrix(adj_hiddens1[0][:,:,ij].detach().cpu().numpy()))
        np.save(args.dataset + '_result/adjs/adj_hidden_'+str(epoch)+'.npy',tt)
 
        if best_score > cv_tra[3]:
            counter += 1
            if counter >= 20:
                break
        else:
            best_epoch=epoch
            best_score = cv_tra[3]
            counter=0

    np.savetxt(args.dataset+'_result/kf_trains_'+str(args.dataset)+'.txt',np.array(cv_trains))  

    data=np.loadtxt(args.dataset + '_result/kf_trains_'+args.dataset+'.txt')
    best_epoch=data[:,3].tolist().index(max(data[:,3].tolist()))
    model = torch.load(args.dataset + '_result/models/model_'+str(best_epoch)+'.pth').cuda()
    
    torch.save(model, args.dataset + '_result/model.pth')
    copyfile(args.dataset + '_result/filter_sub/filter_sub_'+str(best_epoch)+'.npy',args.dataset + '_result/filter_sub_tra.npy')
    copyfile(args.dataset + '_result/filter_sub/sub_node_'+str(best_epoch)+'.npy',args.dataset + '_result/sub_node_tra.npy')
    copyfile(args.dataset + '_result/embeddings/all_local_embeddings_tra_'+str(best_epoch)+'.npy',args.dataset + '_result/all_local_embeddings_tra.npy')
    copyfile(args.dataset + '_result/embeddings/all_global_embeddings_tra_'+str(best_epoch)+'.npy',args.dataset + '_result/all_global_embeddings_tra.npy')
    copyfile(args.dataset + '_result/adjs/adj_hidden_'+str(best_epoch)+'.npy',args.dataset + '_result/adj_hidden.npy')
    
    model.eval()
    _,output_tes1,targets_tes1,cv_tes1,_,el2,eg2,ot2,sn2=preds(MD2,adj_val,features_val,nodes_val,y_val,
                                                      featM_sim,featD_sim,args.batch_size,best_epoch,model,
                                                      optimizer=None,criterion=criterion)
    print(cv_tes1)
    
    np.save(args.dataset + '_result/filter_sub_test.npy',ot2)
    np.save(args.dataset + '_result/sub_node_test.npy',sn2)
    
    tt=[]
    tt=np.concatenate((el2,np.array(targets_tes1).reshape(-1,1)),axis=1)
    np.save(args.dataset + '_result/all_local_embeddings_test.npy',tt)
    tt=[]
    tt=np.concatenate((eg2,np.array(targets_tes1).reshape(-1,1)),axis=1)
    np.save(args.dataset + '_result/all_global_embeddings_test.npy',tt)

    cv_tests.append(comput_metrics(output_tes1,targets_tes1))
    np.savetxt(args.dataset+'_result/kf_tests_'+str(args.dataset)+'.txt',np.array(cv_tes1))
        
    shutil.rmtree(args.dataset + '_result/models')
    shutil.rmtree(args.dataset + '_result/filter_sub')
    shutil.rmtree(args.dataset + '_result/embeddings')
    shutil.rmtree(args.dataset + '_result/adjs')

        