from genericpath import exists
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
import os

parser = argparse.ArgumentParser(description='Training GNN on Paper-Field(L2)  classification task')

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
                    choices=['hgt', 'dense_hgt','hrgat','rgsn','dgat','dhan1','dhan3','v1','v2','v3','v4','v5','v6'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_number', type=int, default=8,
                    help='How many nodes to be sampled per layer per type')
parser.add_argument('--hgt_layer', type=int, default=2,
                    help='Number of HGT layers')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=200,
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
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay of adamw ')

args = parser.parse_args()
set_random_seed(2021)
# if args.cuda != -1:
#     device = torch.device("cuda:" + str(args.cuda))
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")

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

'''
    cand_list stores all the fields, which is the classification domain.
'''
cand_list = list(graph.edge_list['field']['paper']['PFL1'].keys())
'''
Use KL Divergence here, since each paper can be associated with multiple fields.
Thus this task is a multi-label classification.
'''
criterion = nn.KLDivLoss(reduction='batchmean')


'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn = HRGATConv(in_hid=len(graph.node_feature['paper']['emb'].values[0]) + 401, out_hid=args.n_hid, num_m1=6, num_m2=3, \
                conv_name=args.conv_name, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
                hgt_layer=args.hgt_layer).to(device)

classifier = ClassifierT(args.n_hid, len(cand_list)).to(device)

model = nn.Sequential(gnn, classifier)

if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1500, eta_min=1e-6)

stats = []
res = []
best_val = 0
best_loss=100
train_step = 2250

st = time.time()

feature,idl=feature_cal(graph)
class Data():
    def __init__(self):
        super(Data, self).__init__()
        self.train_data = []
        self.valid_data=[]
        self.test_data=[]
        self.in_hid=0
pin=1
rd=0

vdata=dill.Unpickler(open(args.data_dir + '/PF_valid%s.pkl' % args.dataset,'rb')).load()
tdata=dill.Unpickler(open(args.data_dir + '/PF_test%s.pkl' % args.dataset,'rb')).load()
feature,idl=feature_cal(graph)
data = dill.Unpickler(open(args.output_dir + '/sample%s_%s_%s.pkl' % ('PF', args.dataset,pin), 'rb')).load().train_data
for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''

    if epoch !=1 and epoch% 10 ==1:
        pin+=1
        data = dill.Unpickler(open(args.output_dir + '/sample%s_%s_%s.pkl' % ('PF', args.dataset,pin), 'rb')).load().train_data
        rd=0

    train_data = data[rd]
    rd+=1

    '''
        After the data is collected, close the pool and then reopen it.
    '''

    et = time.time()

    '''
        Train (time < 2015)
    '''
    model.train()
    train_losses = []
    torch.cuda.empty_cache()
    for _ in range(args.repeat):
        for paper_gh,author_gh,total_gh,edge_list,x_ids,ylabel in train_data:
            paper_gh=feature_extract(paper_gh,'paper',feature,idl)
            author_gh=feature_extract(author_gh,'author',feature,idl)

            h_a, idl_a, h_p, idl_p = gnn.forward(paper_gh,author_gh,edge_list,device)
     
            id_dict={}
            for idx,idp in enumerate(idl_p):
                id_dict[idp]=idx

            paper_key=[id_dict[id] for id in x_ids ]
            res = classifier.forward(h_p[paper_key])
            
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss
    '''
        Valid (2015 <= time <= 2016)
    '''
    model.eval()
    with torch.no_grad():        
        paper_gh,author_gh,total_gh,edge_list,x_ids,ylabel = vdata
        paper_gh=feature_extract(paper_gh,'paper',feature,idl)
        author_gh=feature_extract(author_gh,'author',feature,idl)

        h_a, idl_a, h_p, idl_p = gnn.forward(paper_gh,author_gh,edge_list,device)

        id_dict={}
        for idx,idp in enumerate(idl_p):
            id_dict[idp]=idx

        paper_key=[id_dict[id] for id in x_ids ]
        res = classifier.forward(h_p[paper_key])
        loss = criterion(res, torch.FloatTensor(ylabel).to(device))
        valid_loss=loss.cpu().detach().tolist()
        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_res = []
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            valid_res += [ai[bi.cpu().numpy()]]
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])

        if valid_ndcg > best_val and valid_loss<=best_loss:
            best_loss=valid_loss
            best_val = valid_ndcg
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')

        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), valid_ndcg))
        del res, loss



'''
    Evaluate 
'''

best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_ndcg=[]
    test_res = []
    test_mrr=[]
    for slice in tdata:
        paper_gh,author_gh,total_gh,edge_list,x_ids,ylabel = slice
        paper_gh=feature_extract(paper_gh,'paper',feature,idl)
        author_gh=feature_extract(author_gh,'author',feature,idl)
        # del total_gh
        # gc.collect()
        h_a, idl_a, h_p, idl_p = gnn.forward(paper_gh,author_gh,edge_list,device)

        id_dict={}
        for idx,idp in enumerate(idl_p):
            id_dict[idp]=idx

        paper_key=[id_dict[id] for id in x_ids ]
        res = classifier.forward(h_p[paper_key])

        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            test_res += [ai[bi.cpu().numpy()]]
        test_ndcg += [ndcg_at_k(resi, len(resi)) for resi in test_res]
        test_mrr += mean_reciprocal_rank(test_res)
    print('Best Test NDCG: %.4f' % np.average(test_ndcg))
    print('Best Test MRR:  %.4f' % np.average(test_mrr))
