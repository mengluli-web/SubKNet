# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:53:19 2022

@author: lml
"""
import argparse
def args_parser():
    parser = argparse.ArgumentParser(description='Models')
    parser.add_argument('--kfs', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='cv',  help='cross-validation (cv) or independent test (indep)')
    parser.add_argument('--epochs', type=int, default=200,  help='number of epochs to train')
    parser.add_argument('--dataset', default='MDA', help='dataset name')
    parser.add_argument('--types', type=str, default='hete', help='homo or hete')

    ### hyper parameters
    parser.add_argument('--lr', type=float, default=1e-2,  help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128,  help='input batch size for training')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate (1 - keep probability).')

    parser.add_argument('--fixs', type=int, default=0,help='the node number of graph filters, Note that 0 represent average of subgraph size in the training set')
    parser.add_argument('--max_step', type=int, default=2, help='max length of random walks')
    parser.add_argument('--gcn_layer', type=int, default=1)
    parser.add_argument('--lambda1', type=float, default=0.5)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--lambda3', type=float, default=0.5)

    args = parser.parse_args()
    return args