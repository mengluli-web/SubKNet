import os 
from parameters import args_parser
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from model import Model
from subgraph_extraction import links2subgraphs
from utils import *
from prediction import preds
import warnings
warnings.filterwarnings('ignore')
seed=1
torch.manual_seed(seed)
np.random.seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
def cv_process():
    args = args_parser()
    if not os.path.exists('result/'):
        os.mkdir('result/')
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

    for iters in range(5):
        idx_train=kfs[iters][0]
        idx_val=kfs[iters][1]

        train_edges=[idx_train_val[i] for i in idx_train]

        val_edges=[idx_train_val[i] for i in idx_val]
        val_positive_edges=np.array([[int(i[0]),int(i[1])] for i in val_edges if i[2]==1])
        val_positive_edges2=np.array([[int(i[1]),int(i[0])] for i in val_edges if i[2]==1])
        val_zero_edges=np.array([[int(i[0]),int(i[1])] for i in val_edges if i[2]==0])

        val_positive_edges = (val_positive_edges.T[0],  val_positive_edges.T[1])
        val_positive_edges2 = (val_positive_edges2.T[0],  val_positive_edges2.T[1])
        val_zero_edges = (val_zero_edges.T[0],  val_zero_edges.T[1])  

        MD = np.copy(data1)
        MD[test_positive_edges] = 0
        MD[val_positive_edges] = 0
        if args.types=='homo':
            MD[test_positive_edges2] = 0
            MD[val_positive_edges2] = 0
            MD2=np.tril(MD, k=0)
        else:
            MD2=MD
        train_positive_edges = np.where(MD2==1)
        train_zero_edges=np.array([[int(i[0]),int(i[1])] for i in train_edges if i[2]==0])#[:len(train_positive_edges[0]),:]
        train_zero_edges = (train_zero_edges.T[0],  train_zero_edges.T[1])

        print(len(train_positive_edges[0]),len(train_zero_edges[0]),len(val_positive_edges[0]),len(val_zero_edges[0]))

        if args.types=='homo':
            MD2=MD2+MD2.T
            featM,featD,featM_sim,featD_sim=comp_feat2(MD2,args.dataset)
        else:    
            featM,featD,featM_sim,featD_sim=comp_feat(MD2,args.dataset)
        
        train_graphs,val_graphs, _, max_n = links2subgraphs(MD2, train_positive_edges,train_zero_edges,
                                                                      val_positive_edges, val_zero_edges,
                                                                      None, None,
                                                                      h=1, max_nodes_per_hop=None,featM=featM,featD=featD)
        n_train = len(train_graphs)
        n_val = len(val_graphs)

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

        cv_trains=[]
        cv_vals=[]
        losses_train=[]
        losses_val=[]
        best_score=0
        counter=0
        for epoch in range(args.epochs):
            model.train()
            loss_tra,output_tra,targets_tra,cv_tra=preds(MD2,adj_train,features_train,nodes_trains,y_train,
                                                          featM_sim,featD_sim,args.batch_size,epoch,model,
                                                          optimizer=optimizer,criterion=criterion)
            print('epoch: ', epoch, loss_tra,'\n',cv_tra)
            cv_trains.append(cv_tra)
            losses_train.append(loss_tra)
            
            model.eval()
            loss_val,output_val,targets_val,cv_val=preds(MD2,adj_val,features_val,nodes_val,y_val,
                                                          featM_sim,featD_sim,args.batch_size,epoch,model,
                                                          optimizer=None,criterion=criterion)
            print('epoch: ', epoch, loss_val,'\n',cv_val)
            cv_vals.append(cv_val)
            losses_val.append(loss_val)
            
            if best_score > cv_val[3]:
                counter += 1
                if counter >= 20:
                    break
            else:
                best_score = cv_val[3]
                counter=0
        np.savetxt('result/kf_trains_'+str(args.dataset)+'_iters_'+str(iters)+'.txt',np.array(cv_trains))
        np.savetxt('result/kf_validations_'+str(args.dataset)+'_iters_'+str(iters)+'.txt',np.array(cv_vals))
