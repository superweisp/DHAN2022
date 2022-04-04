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
# parser.add_argument('--domain', type=str, default='_CS',
#                     help='CS, Medicion or All: _CS or _Med or (empty)')
parser.add_argument('--dataset', type=str, default='c1.0s',
                    help='which dataset to use')
'''
   Model arguments
'''
parser.add_argument('--conv_name', type=str, default='dgat',
                    choices=['dgat'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=128,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=1,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of HeteGAT layers')
parser.add_argument('--dropout', type=float, default=0.1,
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

args = parser.parse_args()

# if args.cuda != -1:
#     device = torch.device("cuda:" + str(args.cuda))
# else:
#     device = torch.device("cpu")
device=torch.device("cpu")
set_random_seed(2021)
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


from torch_geometric.utils import softmax
def mask_softmax(pred, size):
    sz=[]
    ix=[]
    count=0
    for idx,i in enumerate(size):
        sz+=[torch.ones(i)*idx]
        ix+=[count]
        count+=i
    ix=torch.LongTensor(ix)  
    sz=torch.LongTensor(torch.cat(sz).tolist()).to(pred.device)

    loss=torch.log(softmax(pred/0.05,sz))
    loss=loss[ix]/torch.log(torch.FloatTensor(size).to(pred.device))
    return -loss.sum()/len(size)


class Data():
    def __init__(self):
        super(Data, self).__init__()
        self.train_data = []
        self.valid_data=[]
        self.test_data=[]
pin=1
data=dill.Unpickler(open(args.output_dir + '/sample%s_%s_%s.pkl' % ('AD',args.dataset,pin),'rb')).load().train_data

'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn = HRGATConv(in_hid=len(graph.node_feature['paper']['emb'].values[0]) + 401, out_hid=args.n_hid, num_m1=6, num_m2=3, \
                conv_name=args.conv_name, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
                hgt_layer=args.hgt_layer).to(device)

matcher = Matcher(args.n_hid).to(device)

model = nn.Sequential(gnn, matcher)



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
train_step = 2200

best_sta=0
min_loss=10000
feature,idl=feature_cal(graph)
st = time.time()

def rand_mrr(key_size):
    rt=np.average([np.average([1/i for i in range(1,k+1)]) for k in key_size])
    return rt
def rand_ndcg(key_size):
    rt=np.average([np.average(1/np.log2(range(2,i+2))) for i in key_size])
    return rt
round=0
import gc
from collections import defaultdict
log=defaultdict(lambda:[])
record=[]


valid_data=dill.Unpickler(open(args.data_dir + '/AD_valid%s.pkl' % args.dataset,'rb')).load()
tdata=dill.Unpickler(open(args.data_dir + '/AD_test%s.pkl' % args.dataset,'rb')).load()
for epoch in np.arange(args.n_epoch)+1:
    '''
        Prepare Training and Validation Data
    '''

    if epoch !=1 and epoch% 10 ==1:
        pin+=1

        data = dill.Unpickler(open(args.output_dir + '/sample%s_%s_%s.pkl' % ('AD', args.dataset,pin), 'rb')).load().train_data
        round=0

    train_data = data[round]
    round+=1

    et = time.time()

    '''
        Train (time < 2015)
    '''
    model.train()
    train_losses = []
    torch.cuda.empty_cache()
    for _ in range(args.repeat):
        for paper_gh,author_gh,total_gh,edge_list,name_label in train_data:

            
            paper_gh=feature_extract(paper_gh,'paper',feature,idl)
            author_gh=feature_extract(author_gh,'author',feature,idl)
                        
            h_a, idl_a, h_p, idl_p = gnn.forward(paper_gh,author_gh,edge_list,device)
            
            author_key = []
            paper_key = []
            key_size = []

            aid_dict={}
            for idx,aid in enumerate(idl_a):
                aid_dict[aid]=idx
            pid_dict = {}
            for idx,pid in enumerate(idl_p):
                pid_dict[pid] = idx

            for paper_id in name_label:
                author_ids = name_label[paper_id]
                paper_key += [np.repeat(paper_id, len(author_ids))]
                author_key += author_ids
                key_size += [len(author_ids)]
            log['train']+=[key_size]

            paper_key=[pid_dict[pi] for pi in np.concatenate(paper_key)]
            author_key=[aid_dict[ai] for ai in author_key]

            paper_key = torch.LongTensor(paper_key).to(device)
            author_key = torch.LongTensor(author_key).to(device)

            train_paper_vecs = h_p[paper_key]
            train_author_vecs = h_a[author_key]
            
            res = matcher.forward(train_author_vecs, train_paper_vecs, pair=True)

            loss = mask_softmax(res, key_size)

            
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
        
        paper_gh,author_gh,total_gh,edge_list,name_label =valid_data
        paper_gh=feature_extract(paper_gh,'paper',feature,idl)
        author_gh=feature_extract(author_gh,'author',feature,idl)

        h_a, idl_a, h_p, idl_p = gnn.forward(paper_gh,author_gh,edge_list,device)

        author_key = []
        paper_key = []
        key_size = []

        aid_dict = {}
        for aid in idl_a:
            if aid not in aid_dict:
                aid_dict[aid] = len(aid_dict)
        pid_dict = {}
        for pid in idl_p:
            if pid not in pid_dict:
                pid_dict[pid] = len(pid_dict)

        for paper_id in name_label:
            author_ids = name_label[paper_id]
            paper_key += [np.repeat(paper_id, len(author_ids))]
            author_key += [author_ids]
            key_size += [len(author_ids)]
        log['valid']+=[key_size]
        paper_key = [pid_dict[pi] for pi in np.concatenate(paper_key)]
        author_key = [aid_dict[ai] for ai in np.concatenate(author_key)]

        paper_key = torch.LongTensor(paper_key).to(device)
        author_key = torch.LongTensor(author_key).to(device)

        valid_paper_vecs = h_p[paper_key]
        valid_author_vecs = h_a[author_key]

        res = matcher.forward(valid_author_vecs, valid_paper_vecs, pair=True)
        loss = mask_softmax(res, key_size)
        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_res = []
        ser = 0
        for s in key_size:
            p = res[ser: ser + s]
            l = torch.zeros(s)
            l[0] = 1
            r = l[p.argsort(descending=True)]
            valid_res += [r.cpu().detach().tolist()]
            ser += s
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
        valid_mrr = np.average(mean_reciprocal_rank(valid_res))

        valid_acc=np.average(acc(valid_res))
        # rand_res['mrr']+=np.array(key_size)

        if valid_ndcg> best_sta and loss <=min_loss:
            best_sta = valid_ndcg
            min_loss=loss
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')
            record+=['UPDATE!!!']


        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f  Valid MRR: %.4f Valid Acc: %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), valid_ndcg, valid_mrr,valid_acc))
        record+=[("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f  Valid MRR: %.4f Valid Acc: %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), valid_ndcg, valid_mrr,valid_acc)]
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]

        del res, loss


'''
    Evaluate 
'''

best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, matcher = best_model
with torch.no_grad():

    test_res = []
    test_ndcg=[]
    test_mrr=[]
    test_acc=[]
    true_pid = []
    for slice in tdata:
        paper_gh,author_gh,total_gh,edge_list,name_label = slice
        paper_gh=feature_extract(paper_gh,'paper',feature,idl)
        author_gh=feature_extract(author_gh,'author',feature,idl)

        h_a, idl_a, h_p, idl_p = gnn.forward(paper_gh,author_gh,edge_list,device)
        author_key = []
        paper_key = []
        key_size = []

        aid_dict = {}
        for aid in idl_a:
            if aid not in aid_dict:
                aid_dict[aid] = len(aid_dict)
        pid_dict = {}
        for pid in idl_p:
            if pid not in pid_dict:
                pid_dict[pid] = len(pid_dict)


        for paper_id in name_label:
            author_ids = name_label[paper_id]
            true_pid+=[author_ids[0]]
            paper_key += [np.repeat(paper_id, len(author_ids))]
            author_key += [author_ids]
            key_size += [len(author_ids)]
        log['test']+=[key_size]
        paper_key = [pid_dict[pi] for pi in np.concatenate(paper_key)]
        author_key = [aid_dict[ai] for ai in np.concatenate(author_key)]

        paper_key = torch.LongTensor(paper_key).to(device)
        author_key = torch.LongTensor(author_key).to(device)

        test_paper_vecs = h_p[paper_key]
        test_author_vecs = h_a[author_key]
        res = matcher.forward(test_author_vecs, test_paper_vecs, pair=True)

        ser = 0
        for s in key_size:
            p = res[ser: ser + s]
            l = torch.zeros(s)
            l[0] = 1
            r = l[p.argsort(descending=True)]
            test_res += [r.cpu().detach().tolist()]
            ser += s

        test_ndcg += [ndcg_at_k(resi, len(resi)) for resi in test_res]
        test_mrr += mean_reciprocal_rank(test_res)
        test_acc += [acc(test_res)]
    print('Test NDCG: %.4f' % np.average(test_ndcg))
    
    print('Test MRR:  %.4f' % np.average(test_mrr))
    
    print('Test Acc:  %.4f' % np.average(test_acc))
