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

parser = argparse.ArgumentParser(description='Training GNN on Author Disambiguation task')

'''
    Dataset arguments
'''

parser.add_argument('--data_dir', type=str, default='./data',
                    help='The address of preprocessed graph.')
parser.add_argument('--output_dir', type=str, default='./data/c10s',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='AD',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--dataset', type=str, default='c1.0s',
                    help='which dataset to use')
'''
   Model arguments
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'dense_hgt'],
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

# if args.cuda != -1:
#     device = torch.device("cuda:" + str(args.cuda))
# else:
#     device = torch.device("cpu")
# device=torch.device("cpu")

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
#####################################
test_time_bar=2015
adj_author={}
for rel in list(graph.edge_list['author']['paper'].keys()):
    for ad in list(graph.edge_list['author']['paper'][rel].keys()):
        time_list=[]
        for pd in list(graph.edge_list['author']['paper'][rel][ad].keys()):
           t=graph.edge_list['author']['paper'][rel][ad][pd]
           time_list+=[int(t)]
        if not time_list:
            time_list=[0]
        if min(time_list)>=test_time_bar:
            adj_author[ad] =time_list
            del graph.edge_list['author']['paper'][rel][ad]
for rel in list(graph.edge_list['paper']['author'].keys()):
    for pd in list(graph.edge_list['paper']['author'][rel].keys()):
        for ad in list(graph.edge_list['paper']['author'][rel][pd].keys()):
            if ad in adj_author:
                del graph.edge_list['paper']['author'][rel][pd][ad]
##########################################

train_range = {t: True for t in graph.times if t != None and t < 2015}
valid_range = {t: True for t in graph.times if t != None and t >=2015 and t<=2016}
test_range = {t: True for t in graph.times if t != None and t > 2016 }

log['train_range']=train_range
log['valid_range']=valid_range
log['test_range']=test_range

types = graph.get_types()

apd = graph.edge_list['author']['paper']['AP_important']
important_author_dict = {i: True for i in apd if len(apd[i]) >= 2}

name_count = defaultdict(lambda: [])
for i, j in tqdm(graph.node_feature['author'].iterrows(), total=len(graph.node_feature['author'])):
    id=j['id']
    if id in important_author_dict:
        name_count[j['name']] += [id]
name_count = {name: name_count[name] for name in name_count if len(name_count[name]) >= 4}

log['name count number']=len(name_count)



def mask_softmax(pred, size):
    loss = 0
    stx = 0
    for l in size:
        loss += torch.log_softmax(pred[stx: stx + l], dim=-1)[0] / np.log(l)
        stx += l
    return -loss

step=0
step_list={}
from collections import defaultdict
fre_list=defaultdict(lambda :defaultdict(lambda :0))
node_sta={}
total_sta=[]


def author_disambiguation_sample(seed, pairs, time_range, batch_size):
    '''
        sub-graph sampling and label preparation for author disambiguation:
        (1) Sample batch_size // 4 number of names
    '''
    np.random.seed(seed)


    names = np.random.choice(list(pairs.keys()), min(batch_size // 4,len(list(pairs.keys()))), replace=False)

    '''
        (2) Get all the papers written by these same-name authors, and then prepare the label
    '''

    author_dict = {}
    author_info = []
    paper_info = []
    name_label = {}
    max_time = np.max(list(time_range.keys()))


    pl=[]
    al=[]
    gl=[]

    for name in names:
        author_list = name_count[name]
        for a_id in author_list:
            if a_id not in author_dict:
                author_dict[a_id] = len(author_dict)
                author_info += [[a_id, max_time]]
                al+=[a_id]
        for p_id, author_id, _time in pairs[name]:
            paper_info += [[p_id, _time]]
            pl+=[p_id]
            gl+=[author_id]
            '''
                For each paper, create a list: the first entry is the true author's id,
                while the others are negative samples (id of authors with same name)
            '''
            name_label[p_id] = [author_id] + [a_id for a_id in author_list if a_id != author_id]

    global step
    global fre_list
    global node_sta
    global step_list
    global total_sta
    step+=1
    for ai in al:
        fre_list['author'][ai]+=1
    for pi in pl:
        fre_list['paper'][pi]+=1
    step_list[step]=fre_list
    total_sta+=[step_list]

    '''
        (3) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, times, edge_list, indxs, pids= sample_subgraph(graph, time_range, \
                                                      inp={'paper': np.array(paper_info),
                                                           'author': np.array(author_info)}, \
                                                      sampled_depth=args.sample_depth, sampled_number=args.sample_number)
    '''
        (4) Mask out the edge between the output target nodes (paper) with output source nodes (author)
    '''
    true_pair=[]
    for name in names:
        for p in pairs[name]:
            true_pair+=[[p[0],p[1]]]
            true_pair+=[[p[1],p[0]]]



    node_list=defaultdict(lambda :[])
    for t in edge_list:
        for s in edge_list[t]:
            for r in edge_list[t][s]:
                for e in edge_list[t][s][r]:
                    node_list[t]+=[e[0]]
    node_num={}
    for nd in node_list:
        node_num[nd]=len(set(node_list[nd]))
    node_sta[step]=node_num


    masked_edge_list = []

    

    for i in edge_list['paper']['author']['AP_important']:
        if i not in  true_pair:
            masked_edge_list += [i]
    edge_list['paper']['author']['AP_important'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['author']['paper']['AP_important']:
        if i not in  true_pair:
            masked_edge_list += [i]
    edge_list['author']['paper']['AP_important'] = masked_edge_list


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

    '''
        (5) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    # (node_feature, edge_index, edge_type, id_list)
    author_key=[]
    for paper_id in name_label:
        author_ids = name_label[paper_id]
        author_key += author_ids
    author_key=list(set(author_key))

    paper_gh=to_torch(pa_gh)
    author_gh=to_torch(au_gh,idla=author_key)

    

    total_gh={'colleague':ad_sg, 'venue':pvi_sg,'field':pfi_sg,'cite':pd_sg[0],'rev_cite':pd_sg[1],**AP_sg,**PA_sg} 
    total_gh=hgt_to_torch((indxs['paper'],indxs['author']),total_gh,idla=author_key)

    
    return dill.dumps([paper_gh,author_gh,total_gh,edge_list,name_label])


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(author_disambiguation_sample, args=(randint(), \
                                                                 sel_train_pairs, train_range, args.batch_size))
        jobs.append(p)
    p = pool.apply_async(author_disambiguation_sample, args=(randint(), \
                                                             sel_valid_pairs, valid_range, args.batch_size))
    jobs.append(p)
    return jobs


train_pairs = {}
valid_pairs = {}
test_pairs = {}
'''
    Prepare all the author with same name and their written papers.
'''

for name in name_count:
    same_name_author_list = np.array(name_count[name])
    for author_id, author in enumerate(same_name_author_list):
        for p_id in graph.edge_list['author']['paper']['AP_important'][author]:
            _time = graph.edge_list['author']['paper']['AP_important'][author][p_id]
            if type(_time) != int:
                continue
            if _time in train_range:
                if name not in train_pairs:
                    train_pairs[name] = []
                train_pairs[name] += [[p_id, author, _time]]
            elif _time in valid_range:
                if name not in valid_pairs:
                    valid_pairs[name] = []
                valid_pairs[name] += [[p_id, author, _time]]
            elif _time in test_range:
                if name not in test_pairs:
                    test_pairs[name] = []
                test_pairs[name] += [[p_id, author, _time]]



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
log['train pair number']=np.sum([len(sel_train_pairs[name]) for name in sel_train_pairs])
log['valid pair number']=np.sum([len(sel_valid_pairs[name]) for name in sel_valid_pairs])
log['test pair number']=np.sum([len(test_pairs[name]) for name in test_pairs])



pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)
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
    train_data +=[ [dill.loads(job.get()) for job in jobs[:-1]]]
    valid_data +=[ dill.loads(jobs[-1].get())]
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''

    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Epoch : %s Data Preparation: %.1fs' % (epoch,(et - st)))
    st = time.time()
    if epoch % 10 ==0:
        pin += 1
        test_data = []
        for _ in range(10):
            test_data += [dill.loads(author_disambiguation_sample(randint(), test_pairs, test_range, args.batch_size))]
        data = Data()
        data.train_data = train_data
        data.valid_data = valid_data
        data.test_data = test_data
        data.in_hid = len(graph.node_feature['paper']['emb'].values[0])
        dill.dump(data, open(args.output_dir + '/sample%s_%s_%s.pkl' % (args.task_name, args.dataset,pin), 'wb'))
        print('dataset %s has been saved '%pin)
        train_data = []
        valid_data = []




feature,idl=feature_cal(graph)
if not os.path.exists(args.data_dir + '/AD_valid%s.pkl' % args.dataset):
    # tdata=[]
    # for _ in range(10):
    vdata=dill.loads(author_disambiguation_sample(randint(),valid_pairs,valid_range,args.batch_size))
    dill.dump(vdata, open(args.data_dir + '/AD_valid%s.pkl' % args.dataset, 'wb'))
    print('vdata has been saved')
if not os.path.exists(args.data_dir + '/AD_test%s.pkl' % args.dataset):
    tdata=[]
    for _ in range(10):
        tdata+=[dill.loads(author_disambiguation_sample(randint(),test_pairs,test_range,args.batch_size))]
    dill.dump(tdata, open(args.data_dir + '/AD_test%s.pkl' % args.dataset, 'wb'))
    print('tdata has been saved')

