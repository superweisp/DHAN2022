import json, os
import math, copy, time
import numpy as np
from collections import defaultdict
from numpy.lib.function_base import diff
import pandas as pd
from .utils import *
import random
import time

import math
from tqdm import tqdm

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dill
from functools import partial
import multiprocessing as mp

class Graph():
    def __init__(self):
        '''
            node_forward and bacward are only used when building the data.
            Afterwards will be transformed into node_feature by DataFrame

            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        super(Graph, self).__init__()
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])
        '''
            edge_list: index the adjacancy matrix (time) by
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict(  # target_type
            lambda: defaultdict(  # source_type
                lambda: defaultdict(  # relation_type
                    lambda: defaultdict(  # target_id
                        lambda: defaultdict(  # source_id(
                            lambda: int  # time
                        )))))
        self.times = {}
    def add_node(self, node):
        '''
        node is stored as dict: {"type":,"id":,("attr":)}
        '''

        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            self.node_forward[node['type']][node['id']]=len(nfl)
            return node['id']
        return node['id']

    def add_edge(self, target_node, source_node, time=None, relation_type=None,directed = False):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
        self.times[time] = True


    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas

    def get_types(self):
        return list(self.node_feature.keys())


def sample_subgraph(graph, time_range, sampled_depth=2, sampled_number=8, inp=None, feature_extractor=feature_OAG):
    layer_data = defaultdict(  # target_type
        lambda: {}  # {target_id: time}
    )
    budget     = defaultdict( #source_type
                                    lambda: defaultdict(  #source_id
                                        lambda: [0., 0] #[sampled_score, time]
                            ))

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(   # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))
    
    def add_budget(te, target_id, target_time, layer_data, budget):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                if  source_type in ('author') or len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    source_time = adl[source_id]
                    if source_time == None:
                        source_time = target_time
                    if int(source_time) > np.max(list(time_range.keys())) or source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)
                    budget[source_type][source_id][1] = source_time

    '''
        First adding the sampled nodes then updating budget.
    '''
    # inp = {'type': [id,time],}
    for _type in inp:
        for _id, _time in inp[_type]:
            layer_data[_type][_id] = _time
    pids=[]
    aids=[]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            add_budget(te, _id, _time, layer_data, budget)
            if _type=='paper':
                pids+=[_id]
            if _type=='author':
                aids+=[_id]
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys  = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                '''
                    Directly sample all the nodes
                '''
                sampled_ids = np.arange(len(keys))
            elif source_type == 'author':
                # score = np.array(list(budget[target_type][source_type].values()))[:, 0] ** 2
                temp=np.array(list(budget[source_type].values()))[:,0].tolist()
                temp=[float(i) for i in temp]
                score=np.array(temp)**2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), max(sampled_number,int(len(keys)*0.005)), p = score, replace = False)
            else:
                '''
                    Sample based on accumulated degree
                '''
                temp=np.array(list(budget[source_type].values()))[:,0].tolist()
                temp=[float(i) for i in temp]
                score=np.array(temp)**2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]
            for k in sampled_keys:
                add_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)   

    for _type in layer_data:
        for _id in layer_data[_type]:
            edge_list[_type][_type]['self'] += [[_id, _id]]


    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
    '''
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        tld = layer_data[target_type]
        for source_type in te:
            tes = te[source_type]
            sld = layer_data[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_id in tld:
                    if target_id not in tesr:
                        continue
                    target_key = target_id
                    for source_key in tesr[target_key]:
                        '''
                            Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        '''
                        if source_key in sld:
                            source_id = source_key
                            edge_list[target_type][source_type][relation_type] += [[target_id, source_id]]
    for tid in layer_data['author'].keys():
        for sid in layer_data['author'].keys():
            if  set(graph.edge_list['author']['affiliation']['AAf'][sid].keys()) & set(graph.edge_list['author']['affiliation']['AAf'][tid].keys()):
                if tid !=sid:
                    edge_list['author']['author']['colleague'] += [[tid, sid]]
                    edge_list['author']['author']['colleague'] += [[sid, tid]]
    colleague_rel=list(edge_list['author']['author']['colleague'])
    edge_list['author']['author']['colleague']=[colleague_rel[i] for i in np.random.choice(len(colleague_rel), int(len(colleague_rel)/15), replace=False)]

    
    important_paper=list(set(layer_data['paper'].keys())&set(graph.edge_list['paper']['author']['AP_important'].keys()))
    ordinary_paper=list(set(layer_data['paper'].keys())&set(graph.edge_list['paper']['author']['AP_ordinary'].keys()))
    for tid in important_paper:
        for sid in important_paper:
            if  set(graph.edge_list['paper']['author']['AP_important'][tid].keys()) & set(graph.edge_list['paper']['author']['AP_important'][sid].keys()):
                if tid !=sid:
                    edge_list['paper']['paper']['pap1'] += [[tid, sid]]
                    edge_list['paper']['paper']['pap1'] += [[sid, tid]]
    for tid in ordinary_paper:
        for sid in ordinary_paper:
            if  set(graph.edge_list['paper']['author']['AP_ordinary'][tid].keys()) & set(graph.edge_list['paper']['author']['AP_ordinary'][sid].keys()):
                if tid !=sid:
                    edge_list['paper']['paper']['pap2'] += [[tid, sid]]
                    edge_list['paper']['paper']['pap2'] += [[sid, tid]]

    important_author=list(set(layer_data['author'].keys())&set(graph.edge_list['author']['paper']['AP_important'].keys()))
    ordinary_author=list(set(layer_data['author'].keys())&set(graph.edge_list['author']['paper']['AP_ordinary'].keys()))
    for tid in important_author:
        for sid in important_author:
            if  set(graph.edge_list['author']['paper']['AP_important'][tid].keys()) & set(graph.edge_list['author']['paper']['AP_important'][sid].keys()):
                if tid !=sid:
                    edge_list['author']['author']['apa1'] += [[tid, sid]]
                    edge_list['author']['author']['apa1'] += [[sid, tid]]
    for tid in ordinary_author:
        for sid in ordinary_author:
            if  set(graph.edge_list['author']['paper']['AP_ordinary'][tid].keys()) & set(graph.edge_list['author']['paper']['AP_ordinary'][sid].keys()):
                if tid !=sid:
                    edge_list['author']['author']['apa2'] += [[tid, sid]]
                    edge_list['author']['author']['apa2'] += [[sid, tid]]

    feature, times, indxs, texts = feature_extractor(layer_data, graph)
    
    return [], [], edge_list, indxs,pids

from operator import add
from functools import reduce


def meta_subgraph(graph_type,path_type,edge_list,random_edge=False,random_loop=3):
    sub_edge_list={}
    if graph_type=='author' and path_type=='paper':
        sub_edge_list['AP_important']=edge_list['author']['author']['apa1']
        sub_edge_list['AP_ordinary']=edge_list['author']['author']['apa2']
    elif graph_type=='paper' and path_type=='author':
        sub_edge_list['AP_important']=edge_list['paper']['paper']['pap1']
        sub_edge_list['AP_ordinary']=edge_list['paper']['paper']['pap2']
    return sub_edge_list
def indirect_subgraph(graph_type,path_type,edge_list,random_edge=False,random_loop=3,pids=None):
    # if graph_type=='author':
    sub_edge_list = []
    if random_edge:
        candidate_edge = defaultdict(lambda: [])
        for r_type1 in edge_list[graph_type][path_type]:
            for edge_tp in edge_list[graph_type][path_type][r_type1]:
                candidate_edge[edge_tp[1]] += [edge_tp[0]]
        for r_type2 in edge_list[path_type][graph_type]:
            for edge_pt in edge_list[path_type][graph_type][r_type2]:
                candidate_edge[edge_pt[0]]+= [edge_pt[1]]



        subsub_edge = []
        for ce in candidate_edge:
            candidate_edge[ce] = list(set(candidate_edge[ce]))
            for loop in range(random_loop):
                random.shuffle(candidate_edge[ce])
                if len(candidate_edge[ce])==1:
                    subsub_edge+=[[candidate_edge[ce][0],candidate_edge[ce][0]]]
                else:
                    for i in range(len(candidate_edge[ce]) - 1):
                        subsub_edge += [[candidate_edge[ce][i], candidate_edge[ce][i + 1]]]
                        subsub_edge += [[candidate_edge[ce][i + 1], candidate_edge[ce][i]]]
        for e in subsub_edge:
            if e not in sub_edge_list:
                sub_edge_list+=[e]

        if path_type=='field':
            
            new_edge_list=[]
            for _ in range(3):
                sp_pid=pids
                random.shuffle(sub_edge_list)
                for eg in sub_edge_list:
                    if sp_pid :
                        if eg[0] in sp_pid:
                            new_edge_list+=[eg]
                            # sp_pid+=[eg[0]]
                            sp_pid.pop(sp_pid.index(eg[0]))
                        elif eg[1] in sp_pid:
                            new_edge_list+=[eg]
                            sp_pid.pop(sp_pid.index(eg[1]))

            sub_edge_list=[]
            for e in new_edge_list:
                if e not in sub_edge_list:
                    sub_edge_list+=[e]

    else:
        for r1_type in edge_list[graph_type][path_type]:
            for edge_tp in edge_list[graph_type][path_type][r1_type]:
                for r2_type in edge_list[path_type][graph_type]:
                    for edge_pt in edge_list[path_type][graph_type][r2_type]:
                        if edge_tp[1] == edge_pt[0]:
                            sub_edge_list += [[edge_tp[0], edge_pt[1]]]
                            sub_edge_list += [[edge_pt[1], edge_tp[0]]]
        
    return sub_edge_list

def direct_subgraph(edge_list,graph_type,random_edge=False,random_loop=3):
    sub_edge_list = []
    if graph_type=='paper':
        sub_edge_list_PP=[]
        for _edge in edge_list[graph_type][graph_type]['PP_cite']:
            sub_edge_list_PP+= [_edge]
        sub_edge_list_revPP=[]
        for _edge in edge_list[graph_type][graph_type]['rev_PP_cite']:
            sub_edge_list_revPP+= [_edge]
        sub_edge_list+=[sub_edge_list_PP]
        sub_edge_list+=[sub_edge_list_revPP]
    else:
        if random_edge:
            candidate_edge=defaultdict(lambda :[])
            for edge_af in edge_list['author']['affiliation']['AAf']:
                candidate_edge[edge_af[1]]+=[edge_af[0]]
            for edge_fa in edge_list['affiliation']['author']['AAf']:
                candidate_edge[edge_fa[0]] += [edge_fa[1]]

            subsub_edge = []
            for af in candidate_edge:
                candidate_edge[af]=list(set(candidate_edge[af]))
                for loop in range(random_loop):
                    random.shuffle(candidate_edge[af])
                    if len(candidate_edge[af])==1:
                        subsub_edge+=[[candidate_edge[af][0],candidate_edge[af][0]]]
                    else:
                        for i in range(len(candidate_edge[af]) - 1):
                            subsub_edge += [[candidate_edge[af][i], candidate_edge[af][i + 1]]]
                            subsub_edge += [[candidate_edge[af][i + 1], candidate_edge[af][i]]]
            for e in subsub_edge:
                if e not in sub_edge_list:
                    sub_edge_list += [e]


        else:
            sub_edge_list=edge_list['author']['author']['colleague']

    return sub_edge_list


def to_torch(edge_list,idla=None):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''

    edge_index = []
    edge_type = []
    id_list = []
    edge_d={}

    id_index={}
    for _rel in edge_list:
        if _rel not in edge_d.keys():
            edge_d[_rel]=len(edge_d)
        for _edge in edge_list[_rel]:
            if _edge[0] not in id_index.keys():
                id_index[_edge[0]]=len(id_index)
                id_list+=[_edge[0]]
            if _edge[1] not in id_index.keys():
                id_index[_edge[1]] = len(id_index)
                id_list+=[_edge[1]]
            edge_index+=[[id_index[_edge[0]],id_index[_edge[1]]]]
            edge_type+=[edge_d[_rel]]
    if idla:
        for au in idla:
            if au not in id_index.keys():
                edge_d[au]=len(id_index)
                id_list+=[au]

    edge_index = torch.LongTensor(edge_index).t()
    edge_type = torch.LongTensor(edge_type)


    return [[], edge_index, edge_type, id_list]

import numpy as np
def hgt_to_torch(indxs,total_gh,idla=None):
    if idla:
        idx=[indxs[0],list(set(indxs[1].tolist()+idla))]
        indxs=idx
    id_list=np.hstack((indxs[0],indxs[1]))
    node_type=torch.cat((torch.zeros(len(indxs[0])),torch.ones(len(indxs[1]))))
    idd={id:idx for idx,id in enumerate(id_list)}

    edge_index=[]
    edge_dict={et: idx for idx,et in enumerate(total_gh.keys())}
    edge_type=[]
    idx_set=[]
    for gtype in total_gh:
        for eg in total_gh[gtype]:
            edge_index+=[[idd[eg[0]],idd[eg[1]]]]
            edge_type+=[edge_dict[gtype]]
            idx_set+=[idd[eg[0]]]
            idx_set+=[idd[eg[1]]]
    if idla:
        for aid in idla:
            if aid not in idx_set:
                idx_set+=[idd[aid]]

    idx_set=torch.LongTensor(list(set(idx_set)))
    id_list=list(id_list[np.array(idx_set)])
    node_type=node_type[idx_set]

    idx_set=idx_set.tolist()
    id_dict={idx:src for src,idx in enumerate(idx_set)}
    edge_idx=[]
    edge_tp=[]
    for idx,eg in enumerate(edge_index):
        edge_idx+=[[id_dict[eg[0]],id_dict[eg[1]]]]
        edge_tp+=[edge_type[idx]]
    edge_idx=torch.LongTensor(edge_idx).transpose(1,0)
    edge_tp=torch.LongTensor(edge_tp)
    
    return [[], edge_idx, edge_tp, id_list,node_type]






