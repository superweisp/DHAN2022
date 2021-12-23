import sys
from HRGAT.data import *
from HRGAT.model import *
from HRGAT.gats import *
from warnings import filterwarnings
import dill
from HRGAT.data import Graph
filterwarnings("ignore")
import dill as pickle
import argparse

parser = argparse.ArgumentParser(description='Training GNN on Paper-Field(L1)  classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data',
                    help='The address of preprocessed graph.')
parser.add_argument('--output_dir', type=str, default='./data/c10s',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='PF',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--dataset', type=str, default='c1.0s',
                    help='which dataset to use')
'''
   Model arguments
'''
parser.add_argument('--conv_name', type=str, default='hrgat',
                    choices=['hgt', 'dense_hgt','hrgat','rgsn','dgat','dhan1'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=128,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of HeteGAT layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_number', type=int, default=8,
                    help='How many `nodes to be sampled per layer per type')
parser.add_argument('--hgt_layer', type=int, default=1,
                    help='Number of HGT layers')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight decay of adamw ')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=4,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--repeat', type=int, default=2,
                    help='How many time to train over a singe batch (reuse data)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping')

args = parser.parse_args()


log={}


class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "GPT_GNN.data" or module == 'data':
            renamed_module = "HRGAT.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

graph = renamed_load(open(args.data_dir + '/graph_CS_%s.pkl' % args.dataset, 'rb'))


train_range = {t: True for t in graph.times if t != None and t < 2015}
valid_range = {t: True for t in graph.times if t != None and t >=2015 and t<=2016}
test_range = {t: True for t in graph.times if t != None and t > 2016 }



types = graph.get_types()

cand_list = list(graph.edge_list['field']['paper']['PFL1'].keys())




def node_classification_sample(seed, pairs, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers), get their time.
    '''
    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size, replace=False)
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, times, edge_list, indxs, pids = sample_subgraph(graph, time_range, \
                                                              inp={'paper': np.array(target_info)}, \
                                                              sampled_depth=args.sample_depth,
                                                              sampled_number=args.sample_number)

    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (L2 field)
    '''
    masked_edge_list = []
    for i in edge_list['paper']['field']['PFL1']:
        if i[0] not in target_ids:
            masked_edge_list+=[i]
    edge_list['paper']['field']['PFL1'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['field']['paper']['PFL1']:
        if i[1] not in target_ids:
            masked_edge_list+=[i]
    edge_list['field']['paper']['PFL1'] = masked_edge_list

    '''
        (4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''

    api_sg=meta_subgraph('author', 'paper', edge_list,random_edge=True)
    ad_sg=direct_subgraph(edge_list, 'author',random_edge=True)

    pvi_sg=indirect_subgraph('paper', 'venue', edge_list,random_edge=True)
    pfi_sg = indirect_subgraph('paper', 'field', edge_list,random_edge=True,random_loop=1,pids=pids)
    pai_sg = meta_subgraph('paper', 'author', edge_list,random_edge=True)
    pd_sg=direct_subgraph(edge_list, 'paper',random_edge=True)#two directed graphs
    ############################
    au_gh={'colleague':ad_sg,'apa1':api_sg['AP_important'],'apa2':api_sg['AP_ordinary'],}
    pa_gh={'venue':pvi_sg,'field':pfi_sg,'pap1':pai_sg['AP_important'],'pap2':pai_sg['AP_ordinary'],'cite':pd_sg[0],'rev_cite':pd_sg[1]}

    AP_sg={'AP1':edge_list['author']['paper']['AP_important'],'AP2':edge_list['author']['paper']['AP_ordinary']}
    PA_sg={'PA1':edge_list['paper']['author']['AP_important'],'PA2':edge_list['paper']['author']['AP_ordinary']}

    # paper_gh=to_torch(feature['paper'], indxs['paper'], pa_gh)
    # author_gh=to_torch(feature['author'], indxs['author'], au_gh)
    paper_gh=to_torch(pa_gh)
    author_gh=to_torch(au_gh)

    total_gh={'colleague':ad_sg, 'venue':pvi_sg,'field':pfi_sg,'cite':pd_sg[0],'rev_cite':pd_sg[1],**AP_sg,**PA_sg} 
    # total_gh=hgt_to_torch((feature['paper'],feature['author']),(indxs['paper'],indxs['author']),total_gh)
    total_gh=hgt_to_torch((indxs['paper'],indxs['author']),total_gh)
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    ylabel = np.zeros([args.batch_size, len(cand_list)])
    for x_id, target_id in enumerate(target_ids):
        for source_id in pairs[target_id][0]:
            ylabel[x_id][cand_list.index(source_id)] = 1
    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
    x_ids = list(target_ids)
    
    return [paper_gh,author_gh,total_gh,edge_list,x_ids,ylabel]


def prepare_data():
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = node_classification_sample(randint(),sel_train_pairs, train_range)
        jobs.append(p)
    p = node_classification_sample(randint(), sel_valid_pairs, valid_range)
    jobs.append(p)
    return jobs


train_pairs = {}
valid_pairs = {}
test_pairs = {}

for target_id in graph.edge_list['paper']['field']['PFL1']:
    for source_id in graph.edge_list['paper']['field']['PFL1'][target_id]:
        _time = graph.edge_list['paper']['field']['PFL1'][target_id][source_id]
        if _time in train_range:
            if target_id not in train_pairs:
                train_pairs[target_id] = [[], _time]
            train_pairs[target_id][0] += [source_id]
        elif _time in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [[], _time]
            valid_pairs[target_id][0] += [source_id]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id] = [[], _time]
            test_pairs[target_id][0] += [source_id]



np.random.seed(43)
'''
    Only train and valid with a certain percentage of data, if necessary.
'''
sel_train_pairs = {p: train_pairs[p] for p in
                   np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage),
                                    replace=False)}
sel_valid_pairs = {p: valid_pairs[p] for p in
                   np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage),
                                    replace=False)}

st = time.time()
jobs = prepare_data()
train_data=[]
valid_data=[]
import gc
class Data():
    def __init__(self):
        super(Data, self).__init__()
        self.train_data = []
        self.valid_data=[]
        self.test_data=[]
        self.in_hid=0

pin=0
for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    train_data +=[jobs[:-1]]
    valid_data +=[jobs[-1]]
 
    '''
        After the data is collected, close the pool and then reopen it.
    '''

    jobs = prepare_data()
    et = time.time()
    print('Epoch : %s Data Preparation: %.1fs' % (epoch,(et - st)))
    st = time.time()
    if epoch % 10 ==0:
        pin += 1
        test_data = []
        for _ in range(10):
            test_data += [node_classification_sample(randint(),test_pairs,test_range)]
        data = Data()
        data.train_data = train_data
        data.valid_data = valid_data
        data.test_data = test_data
        data.in_hid = len(graph.node_feature['paper']['emb'].values[0])
        dill.dump(data, open(args.output_dir + '/sample%s_%s_%s.pkl' % (args.task_name, args.dataset,pin), 'wb'))
        print('dataset %s has been saved '%pin)
        train_data = []
        valid_data = []



jobs = prepare_data()

feature,idl=feature_cal(graph)
if not os.path.exists(args.data_dir + '/PF_valid%s.pkl' % args.dataset):
    tdata=[]
    vdata=node_classification_sample(randint(),valid_pairs,valid_range)
    dill.dump(vdata, open(args.data_dir + '/PF_valid%s.pkl' % args.dataset, 'wb'))
    print('vdata has been saved')
if not os.path.exists(args.data_dir + '/PF_test%s.pkl' % args.dataset):
    tdata=[]
    for _ in range(10):
        tdata+=[node_classification_sample(randint(),test_pairs,test_range)]
    dill.dump(tdata, open(args.data_dir + '/PF_test%s.pkl' % args.dataset, 'wb'))
    print('tdata has been saved')