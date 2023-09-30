import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import DenseGCNConv
import numpy as np
seed=1
torch.manual_seed(seed)
np.random.seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
from torch.autograd import Variable  
import scipy.sparse as sp  
def MyPrepareSparseMatrices(graph_list, graph_sizes):
    account = 0
    all_node_num = graph_sizes[0]*graph_list.shape[0]
    n2n = np.zeros((all_node_num, all_node_num))
    for i in range(len(graph_list)):
        ep_pos = (np.where(graph_list[i]==1)[0]+account, 
                  np.where(graph_list[i]==1)[1]+account)
        n2n[ep_pos] = 1;  n2n[(ep_pos[1],ep_pos[0])] = 1
        account += graph_sizes[i]
    n2n_sp = sp.coo_matrix(n2n)
    values = n2n_sp.data
    indices = np.vstack((n2n_sp.row, n2n_sp.col))
    values = torch.FloatTensor(values)
    indices = torch.LongTensor(indices)
    shape = n2n_sp.shape

    return torch.sparse.FloatTensor(indices, values, shape)

class RW_layer(nn.Module):  
    def __init__(self, input_dim, out_dim,hidden_dim = None, max_step = 1, size_graph_filter = 10, dropout = 0.5):
        super(RW_layer, self).__init__()
        self.max_step = max_step
        self.size_graph_filter = size_graph_filter
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.fc_in = torch.nn.Linear(input_dim, hidden_dim)
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, hidden_dim, out_dim))
        else:
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, input_dim, out_dim))
        self.adj_hidden = Parameter(torch.FloatTensor( (size_graph_filter*(size_graph_filter-1))//2 , out_dim))
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        
    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
        
    def forward(self, adj, features):
        adj_hidden=self.adj_hidden.cuda()
        adj_hidden = torch.clamp(self.adj_hidden.cuda(), min=-1, max=1)
        adj_hidden_norm = torch.zeros( self.size_graph_filter, self.size_graph_filter, self.out_dim).cuda()
        idx = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1).cuda()
        adj_hidden_norm[idx[0], idx[1], :] = self.relu(adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)
        z = self.features_hidden.cuda() # (Nhid,Dhid,Dout)
        z = torch.clamp(z, min=0, max=1)
        feat_hidds=z
        #construct feature array for each subgraph
        x=features
        # x = features
        if self.hidden_dim:
            x = nn.ReLU()(self.fc_in(x)) # (#G, D_hid)
        zx = torch.einsum("mcn,abc->ambn", (z, x)) # (#G, #Nodes_filter, #Nodes_sub, D_out)
        
        output2 = []
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.size_graph_filter).cuda()            
                o = torch.einsum("ab,bcd->acd", (eye, z))
                t = torch.einsum("mcn,abc->ambn", (o, x))
            else:
                x = torch.einsum("abc,acd->abd",(adj, x))
                z = torch.einsum("abd,bcd->acd", (adj_hidden_norm, z)) # adj_hidden_norm: (Nhid,Nhid,Dout)
                t = torch.einsum("mcn,abc->ambn", (z, x))
            
            t = self.dropout(t) 
            t = torch.mul(zx, t) # (#G, #Nodes_filter, #Nodes_sub, D_out)
            t = torch.mean(t, dim=[1,2])
            output2.append(t)
        output2 = sum(output2)/len(output2)
        out=output2
        #return out,adj_hidden_norm,feat_hidds
        L_diversity = 0.5*self._diversity_term(torch.mean(feat_hidds.permute(2,0,1), dim=[1]))+0.5*self._diversity_term(adj_hidden_norm.permute(2,0,1).flatten(1))
        #print(L_diversity)
        return out,L_diversity 

    def _diversity_term(self, x, d="euclidean", eps=1e-9):
        D = torch.cdist(x.contiguous(), x.contiguous(), 2)
        Rd = torch.relu(-D + 2)

        zero_diag = torch.ones_like(Rd, device=Rd.device) - torch.eye(
            x.shape[-2], device=Rd.device
        )
        return ((Rd * zero_diag)).sum() / 2.0
        
                
class Model(nn.Module):
    def __init__(self,m_size,d_size,input_dim, output_dim,args,hidden_dims = [16,32], 
                 size_graph_filter = None, max_step = 1, 
                 dropout_rate=0.5, size_subgraph = None):
        super(Model, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_layers = len(hidden_dims)-1
        self.ker_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.ker_layers.append(RW_layer(input_dim, hidden_dims[1], hidden_dim = hidden_dims[0], 
                                                max_step = max_step, size_graph_filter = size_graph_filter[0], 
                                                dropout = dropout_rate))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[1]))
            else:
                self.ker_layers.append(RW_layer(hidden_dims[layer+1], hidden_dims[layer], hidden_dim = hidden_dims[layer], 
                                                max_step = max_step, size_graph_filter = size_graph_filter[0], 
                                                dropout = dropout_rate))                          
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[layer+1]))

        self.gcn_layers = torch.nn.ModuleList()
        for layer in range(args.gcn_layer):
            if layer == 0:
                self.gcn_layers.append(DenseGCNConv(64,16))
            else:
                self.gcn_layers.append(DenseGCNConv(16,16))

        self.fcm = torch.nn.Linear(m_size, 64)
        self.bnm = nn.BatchNorm1d(m_size)
        self.dropoutm = nn.Dropout(p=dropout_rate)
        
        self.fcd = torch.nn.Linear(d_size, 64)
        self.bnd = nn.BatchNorm1d(d_size)
        self.dropoutd = nn.Dropout(p=dropout_rate)
        self.fcs = torch.nn.Linear(64, 32)
        self.fcs2 = torch.nn.Linear(32, output_dim)
        self.fcs_srl = torch.nn.Linear(32, output_dim)
        self.fcs_grl = torch.nn.Linear(32, output_dim)
        
        self.gnn = DenseGCNConv(input_dim, 32)
        latent_dim=[32, 32, 32, 1]
        k=30
        conv1d_channels=[16, 32]
        conv1d_kws=[0, 5]
        conv1d_activation='ReLU'
        num_node_feats=input_dim
        self.latent_dim = latent_dim
        self.output_dim = 32
        self.num_node_feats = input_dim
        self.k = k
        self.total_latent_dim = 32
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, self.output_dim)

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))
        
    def forward(self, mds, adj, h,mfeat,dfeat,sub_nodes,args): 
        for layer in range(self.num_layers):
            h,los2= self.ker_layers[layer](adj, h)#,adj_hidd,feat_hidd
            if torch.sum(h)!=0:
                h = self.batch_norms[layer](h)
            h = F.relu(h)
        outs_srl=self.fcs_srl(h)
        outs_srl=F.softmax(outs_srl, dim=1)

        if args.types=='homo':
            Y_m=torch.FloatTensor(mfeat).cuda()
            m_embedding=self.bnm(Y_m)
            m_embedding=self.fcm(m_embedding)
            m_embedding=F.relu(m_embedding)
            m_embedding=self.dropoutm(m_embedding)

            h2=m_embedding
            for layer in range(args.gcn_layer):
                h2 = self.gcn_layers[layer](h2,mds)[0]
                h2 = F.relu(h2)
            h2=torch.cat((h2[sub_nodes[:,0]],h2[sub_nodes[:,1]]),1)#
        else:
            Y_m=torch.FloatTensor(mfeat).cuda()
            Y_d=torch.FloatTensor(dfeat).cuda()
            # rrs=0
            m_embedding=self.bnm(Y_m)
            m_embedding=self.fcm(m_embedding)
            m_embedding=F.relu(m_embedding)
            m_embedding=self.dropoutm(m_embedding)
            
            d_embedding=self.bnd(Y_d)
            d_embedding=self.fcd(d_embedding)
            d_embedding=F.relu(d_embedding)
            d_embedding=self.dropoutd(d_embedding)
            m_d_embedding=torch.cat((m_embedding,d_embedding),0)
            h2=m_d_embedding
            for layer in range(args.gcn_layer):
                h2 = self.gcn_layers[layer](h2,mds)[0]
                h2 = F.relu(h2)
            h2=torch.cat((h2[sub_nodes[:,0]],h2[mfeat.shape[0]+sub_nodes[:,1]]),1)#
        outs_grl=self.fcs_grl(h2)
        outs_grl=F.softmax(outs_grl, dim=1)
        preds=self.fcs(torch.cat((h,h2),1))
        preds=F.relu(preds)
        preds=self.fcs2(preds)
        outs=F.softmax(preds, dim=1)
        return outs,los2,outs_srl,outs_grl
       
