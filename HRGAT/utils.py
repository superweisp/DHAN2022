import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict

# def dcg_at_k(r, k):
#     r = np.asfarray(r)[:k]
#     if r.size:
#         return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
#     return 0.


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return  np.sum(r/ np.log2(np.arange(2, r.size + 2)))
    return 0.




def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]



def acc(rs):
    total=0
    correct=0
    for r in rs:
        if r[0]==1:
            correct+=1
        total+=1
    return correct/total


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # "input:mxn,output:indicees-->2xMax(m,n),values-->t(number of items that is not 0),shape-->mxn"
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# def randint():
#     return np.random.randint(2**32 - 1)




def feature_OAG(layer_data, graph):
    feature = {}
    times = {}
    indxs = {}
    texts = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs = np.array(list(layer_data[_type].keys()))

        indxs[_type] = idxs
    return feature, times, indxs, texts

def feature_cal(graph):
    feature = {}
    idl={}
    for _type in ['paper','author']:
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type]['node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(graph.node_feature[_type]), 400])

        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type]['emb']), \
                                         np.log10(np.array(list(graph.node_feature[_type]['citation'])).reshape(-1,1) + 0.01)),
                                        axis=1)
        # idl[_type]=graph.node_feature[_type]['id']
        idx_dict={}
        for idx,id in enumerate(graph.node_feature[_type]['id']):
            idx_dict[id]=idx
        idl[_type]=idx_dict
    return feature,idl

# _gh=(node_feature, edge_index, edge_type, id_list)
def feature_extract(gh,type,feature,idl):
    id_list=gh[3]
    idx=[]
    for i in id_list:
        idx+=[idl[type][i]]
    idx=np.array(idx)
    node_feature=torch.FloatTensor(feature[type][idx].tolist())
    # gh[0]=node_feature
    return (node_feature,gh[1],gh[2],gh[3])
#gh:([], edge_idx, edge_tp, id_list,node_type)
#node_type:{'paper':0,"author":1}
def hgt_extract(gh,feature,idl):
    paper_num=(gh[4]==0).sum()
    paper_idl=gh[3][:paper_num]
    author_idl=gh[3][paper_num:]
    p_idx=[]
    for i in paper_idl:
        p_idx+=[idl['paper'][i]]
    a_idx=[]
    for i in author_idl:
        a_idx+=[idl['author'][i]]
    p_idx=np.array(p_idx)
    a_idx=np.array(a_idx)
    node_feature=np.concatenate((feature['paper'][p_idx],feature['author'][a_idx]),axis=0)
    node_feature=torch.FloatTensor(node_feature)
    # gh[0]=node_feature
    return (node_feature,gh[1],gh[2],gh[3],gh[4])
        






def rel_graph(idl_a,idl_p,subgraph):
    ad={}
    pd={}
    ed={}
    edge_indx=[]
    edge_type=[]
    for idx,an in enumerate(idl_a):
        ad[an]=idx
    for idx,an in enumerate(idl_p):
        pd[an] = idx

    for rel in subgraph["author"]["paper"]:
        if rel not in ed.keys():
            ed[rel]=len(ed)
        for _edge in subgraph["author"]["paper"][rel]:
            if _edge[0] in idl_a and _edge[1] in idl_p:
                edge_indx+=[[ad[_edge[0]],pd[_edge[1]]]]
                edge_type+=[ed[rel]]

    edge_type = torch.LongTensor(edge_type)
    return torch.LongTensor(edge_indx).t(),edge_type


def randint():
    return np.random.randint(2**32 - 1)

import os
import random
def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
