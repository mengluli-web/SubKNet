# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:14:29 2023

@author: lml
"""
import torch
import time
import scipy.sparse as sp
import numpy as np
from parameters import args_parser
seed=1
torch.manual_seed(seed)
np.random.seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
import random
import scipy.sparse as ssp

def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

def getNeighborsFromM(fr0, fr2, Mtx) -> set:
    pure_up = set()
    mixed = set()
    # direct
    if type(Mtx) is not np.matrix:
        Mtx = np.matrix(Mtx)
    if fr2 is None:
        for n in fr0:
            pure_up_ctx = np.where(Mtx[n,:] == 1)[1]
            pure_up_ctx = set(pure_up_ctx)
            pure_up = pure_up.union(pure_up_ctx)
            return pure_up, mixed
    # search in all '+' link
    for n in fr0:
        pure_up_ctx = np.where(Mtx[n,:] == 1)[1]
        pure_up = pure_up.union(pure_up_ctx)
    # mixed
    for n in fr2:
        mixed_ctx = np.where(Mtx[n,:] != 0)[1]
        mixed = mixed.union(mixed_ctx)
    # pure link first
    mixed = mixed - (mixed&pure_up)
    pure_up  = pure_up - mixed
    return pure_up, mixed

def getNeighborsFromD(fr0, fr2, Mtx) -> set:
    pure_up = set()
    mixed = set()

    if type(Mtx) is not np.matrix:
        Mtx = np.matrix(Mtx)
    if fr2 is None:
        for n in fr0:
            pure_up_ctx = np.where(Mtx[:,n] == 1)[0]
            pure_up = pure_up.union(pure_up_ctx)
            return pure_up, mixed
    for n in fr0:
        pure_up_ctx = np.where(Mtx[:,n] == 1)[0]
        pure_up = pure_up.union(pure_up_ctx)
    for n in fr2:
        mixed_ctx = np.where(Mtx[:,n] != 0)[0]
        mixed = mixed.union(mixed_ctx)

    mixed = mixed - (mixed&pure_up)
    pure_up  = pure_up - mixed
    return pure_up, mixed

def constructNet(miRNA_disease=None, miRNA_similarity=None, disease_similarity=None):
    if miRNA_disease is None:
        # Special purpose: unless you know what you are doing, don't let this parameter be None.
        miRNA_disease = np.zeros((miRNA_similarity.shape[0], disease_similarity.shape[0]))
    if miRNA_similarity is None:
        miRNA_similarity = np.array(np.eye(miRNA_disease.shape[0]), dtype=np.int8)
    if disease_similarity is None:
        disease_similarity = np.array(np.eye(miRNA_disease.shape[1]), dtype=np.int8)
    m1 = np.hstack((miRNA_similarity, miRNA_disease))
    m2 = np.hstack((miRNA_disease.T, disease_similarity))
    return np.vstack((m1, m2))

def subgraph_extraction_labeling(ind, Mtx:np.matrix,h=1, max_nodes_per_hop=None, featM=None, featD=None,g_label=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    m_nodes, d_nodes = [], []
    m_0_visited, m_2_visited = set([]), set([])
    d_0_visited, d_2_visited = set([]), set([])
    
    m_0_fringe, m_2_fringe = set([]), set([])
    d_0_fringe, d_2_fringe = set([]), set([])

    m_0_mark, m_2_mark = set([]), set([])
    d_0_mark, d_2_mark = set([]), set([])
    node_label_m, node_label_d = [], []  ## [0],  [1]
    
    # node labeling
    for hop in range(1, h+1):
        if hop == 1:
            d_0_fringe, d_2_fringe = getNeighborsFromM(set([ind[0]]),None, Mtx)
            m_0_fringe, m_2_fringe = getNeighborsFromD(set([ind[1]]),None, Mtx)
        else:
            d_0_fringe, d_2_fringe = getNeighborsFromM(m_0_mark, m_2_mark, Mtx)
            m_0_fringe, m_2_fringe = getNeighborsFromD(d_0_mark, d_2_mark, Mtx)
        assert not bool(d_0_fringe&d_2_fringe), "ERROR: found repeat number in hop {}".format(hop)
        assert not bool(m_0_fringe&m_2_fringe), "ERROR: found repeat number in hop {}".format(hop)
        m_0_fringe = m_0_fringe - m_0_visited - m_2_visited - set(m_nodes)
        m_2_fringe = m_2_fringe - m_0_visited - m_2_visited - set(m_nodes)
        d_0_fringe = d_0_fringe - d_0_visited - d_2_visited - set(d_nodes)
        d_2_fringe = d_2_fringe - d_0_visited - d_2_visited - set(d_nodes)
        m_0_mark = m_0_fringe  # pred result
        m_2_mark = m_2_fringe  # pred result
        d_0_mark = d_0_fringe  # pred result
        d_2_mark = d_2_fringe  # pred result
        # update visit
        m_0_visited = m_0_visited.union(m_0_fringe)
        m_2_visited = m_2_visited.union(m_2_fringe)
        d_0_visited = d_0_visited.union(d_0_fringe)
        d_2_visited = d_2_visited.union(d_2_fringe)
        # ignore none link
        if (len(m_0_fringe) == 0 and len(m_2_fringe) == 0 
            and len(d_0_fringe) == 0 and len(d_2_fringe) == 0):
            break
        m_nodes = m_nodes + list(m_0_fringe) + list(m_2_fringe)
        d_nodes = d_nodes + list(d_0_fringe) + list(d_2_fringe)
        assert len(m_nodes) == len(set(m_nodes)), "Error: Duplicate node found in hop {}.".format(hop)
        assert len(d_nodes) == len(set(d_nodes)), "Error: Duplicate node found in hop {}.".format(hop)
        node_label_m = node_label_m + [4*hop-2] * len(m_0_fringe) + [4*hop]   * len(m_2_fringe)
        node_label_d = node_label_d + [4*hop-1] * len(d_0_fringe) + [4*hop+1] * len(d_2_fringe)
        print('',end='')
    rand_state_m = np.random.get_state()
    np.random.set_state(rand_state_m); np.random.shuffle(m_nodes)
    np.random.set_state(rand_state_m); np.random.shuffle(node_label_m)
    rand_state_d = np.random.get_state()
    np.random.set_state(rand_state_d); np.random.shuffle(d_nodes)
    np.random.set_state(rand_state_d); np.random.shuffle(node_label_d)
    if ind[0] in m_nodes:
        del node_label_m[m_nodes.index(ind[0])]
        m_nodes.remove(ind[0])
    if ind[1] in d_nodes:
        del node_label_d[d_nodes.index(ind[1])]
        d_nodes.remove(ind[1])
        
    args = args_parser()
    if args.types=='homo':
        d_nodes_new=[i for i in d_nodes if i not in m_nodes]
        d_nodes=d_nodes_new
        m_nodes = [ind[0]] + m_nodes
        d_nodes=[ind[1]] + d_nodes
        mm=m_nodes+d_nodes
        node_label_m = [0] + node_label_m
        node_label_d = [1] + node_label_d
        node_labels = np.array(node_label_m + node_label_d)
        # print(m_nodes,d_nodes,mm,node_label_m,node_label_d)
        subgraph = np.array(Mtx[mm, :][:, mm])
        subgraph[0, len(m_nodes)] = 0
        subgraph[len(m_nodes), 0] = 0
        # print(subgraph)
        subgraphAdj = np.array(subgraph)#+np.eye(subgraph.shape[0])
        node_feat=np.vstack((featM[mm,:]))
        # print(subgraphAdj)
    else:
        m_nodes = [ind[0]] + m_nodes
        d_nodes = [ind[1]] + d_nodes
        node_label_m = [0] + node_label_m
        node_label_d = [1] + node_label_d
        # print(m_nodes,d_nodes)
        subgraph = np.array(Mtx[m_nodes, :][:, d_nodes])
        node_labels = np.array(node_label_m + node_label_d)
        # print(subgraph)
        subgraph[0, 0] = 0
        
        # generate graph
        subgraphAdj = constructNet(subgraph)
        # print(subgraphAdj)
        node_feat=np.vstack((featM[m_nodes,:],featD[d_nodes,:]))

    return sp.coo_matrix(subgraphAdj),sp.coo_matrix(node_feat),[ind[0],ind[1]],g_label,node_labels,max(node_labels),m_nodes,d_nodes


def links2subgraphs(Mtx, train_pos,train_zero, val_pos, val_zero, test_pos, test_zero, h=1, max_nodes_per_hop=None, featM=None,featD=None) -> list:
    # extract enclosing subgraphs
    ###############################################################
    global train_max_shape
    global val_max_shape
    def helper(M, links, g_label):
        global max_n
        start = time.time()

        results = [subgraph_extraction_labeling(*((i, j), M, h, max_nodes_per_hop, featM, featD,g_label)) for i, j in zip(links[0], links[1])]
        if len(results) != 0:
            max_n = max([i[-1] for i in results])
        end = time.time()
        print(" \rSubgraph extracting ... ok (Time: {:.2f}s)".format(end-start), flush=True)
        return results
    ###########################################################
    if train_pos: 
        train_graphs = helper(Mtx, train_pos, 1) + helper(Mtx, train_zero, 0)
    else:
        train_graphs=[]
    if test_pos: 
        test_graphs = helper(Mtx, test_pos, 1) + helper(Mtx, test_zero, 0)
    else:
        test_graphs=[]
    if val_pos: 
        val_graphs = helper(Mtx, val_pos, 1) + helper(Mtx, val_zero, 0)
    else:
        val_graphs=[]
    
    return train_graphs, val_graphs,test_graphs, max_n