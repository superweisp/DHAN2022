# from _typeshed import Self
from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv,RGCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math
from .utils import rel_graph
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn import Sequential, Linear, ReLU, Dropout




class DGATConv(MessagePassing):
    def __init__(self, in_hid, out_hid, 
                 num_edge_types,negative_slope=0.2,dual=True,heads=1,mask=None,global_weight=True):
        super(DGATConv, self).__init__(aggr='add')

        self.in_hid = in_hid
        self.out_hid = out_hid
        self.num_edge_types = num_edge_types
        self.negative_slope=negative_slope
        self.dual=dual
        self.mask=mask
        self.global_weight=global_weight
        
        self.rel_wi=nn.Parameter(torch.Tensor(num_edge_types,out_hid*2,1))

        self.rel_bt=nn.Parameter(torch.Tensor(out_hid*2,1))
        self.w_wi=nn.Linear(in_hid, out_hid, bias=False)
        self.w_bt=nn.Linear(out_hid,out_hid,bias=False)

        self.w_out=nn.Linear(out_hid,out_hid,bias=False)
        self.q_trans=nn.Parameter(torch.Tensor(out_hid,1))

        self.norm=nn.LayerNorm(out_hid)
        self.norm_list=nn.ModuleList()
        for i in range(num_edge_types):
            self.norm_list.append(nn.LayerNorm(out_hid))


        self.skip = nn.Parameter(torch.ones(1))
        self.beta_weight=nn.Parameter(torch.ones(1))
        self.overall_beta=nn.Parameter(torch.randn(num_edge_types))
        # self.drop=Dropout(0.2)

        glorot(self.rel_wi)
        glorot(self.rel_bt)
        glorot(self.q_trans)



    def forward(self, x, edge_idx, edge_type):

        x=self.w_wi(x)
        out_list=[]
        edg_list=[]
        overall_rel=[]
        for i in range(self.num_edge_types):
            mask = (edge_type == i)
            edge_index = edge_idx[:, mask]
            if mask.sum() !=0:
                rs=self.w_bt(F.leaky_relu(self.norm_list[i](self.propagate(edge_index, x=x,edge_type=i)),self.negative_slope))   #Nxd
                out_list+=[rs]
                edg_list+=[i]
     
            
        if self.dual:
            overall_beta=F.softmax(self.overall_beta,dim=0)

            rs_list=[]
            for i in range(len(edg_list)):
                conc=torch.cat((x,out_list[i]),dim=1)                                       #Nx2d
                rs=torch.matmul(conc,self.rel_bt)         #Nx1
                rs_list+=[rs]

            rs=torch.stack(rs_list)                                                         #rxNx1
            beta=F.softmax(rs,dim=0)                                                        #rxNx1
            res=0
            if self.mask:
                for i in self.mask:
                    out_list[i]=torch.zeros_like(out_list[i])
            beta_weight=torch.sigmoid(self.beta_weight)
            for i in range(len(edg_list)):
                if self.global_weight:
                    res+=out_list[i]*((1-beta_weight)*beta[i]+beta_weight*overall_beta[i])
                else:
                    res+=out_list[i]*beta[i]
        else:
            res=0
            for i in range(len(edg_list)):
                res+=out_list[i]
        
        final_weight=torch.sigmoid(self.skip)
        res = self.norm(F.gelu(res) * (final_weight) + x* (1 - final_weight))

        return res


    def message(self,edge_index,x_i, x_j,edge_type):
        
        node_f = torch.cat((x_i, x_j), 1)                                       #nx2d

        temp = torch.matmul(node_f, self.rel_wi[edge_type]).to(x_i.device)      #nx1

        alpha=softmax(temp,edge_index[1])

        rs=x_j*alpha                                                            #nxd
        return rs


sft = torch.nn.Softmax(dim=0)


class HetGATConv(MessagePassing):
    def __init__(self, in_hid, out_hid, negative_slope=0.2,norm=True,dual=True,global_weight=True):
        super(HetGATConv, self).__init__(aggr='add')

        self.in_hid = in_hid
        self.out_hid = out_hid
        self.negative_slope=negative_slope
        self.norm=norm
        self.dual=dual
        self.global_weight=global_weight

        
        self.rel_wi=nn.Parameter(torch.Tensor(2,out_hid*2,1))
        self.rel_bt=nn.Parameter(torch.Tensor(out_hid*2,1))
        self.w_bt=nn.Linear(out_hid,out_hid,bias=False)
        self.w_out=nn.Linear(out_hid,out_hid,bias=False)

        self.out_norm=nn.LayerNorm(out_hid)

        self.skip = nn.Parameter(torch.ones(1))

        glorot(self.rel_wi)
        glorot(self.rel_bt)
        

    def forward(self, a_hid,p_hid, edge_idx, edge_type):

        xi=p_hid[edge_idx[1]]
        out_list=[]
        num_edge_types=2
        edg_list=[]
        for i in range(num_edge_types):
            mask = (edge_type == i)
            edge_index = edge_idx[:, mask]
            if mask.sum() !=0:
                rs=self.w_bt(F.leaky_relu(self.propagate(edge_index, x=(a_hid,p_hid),edge_type=i),self.negative_slope))   #Nxd
                out_list+=[rs]
                edg_list+=[i]
            
        if self.dual:
            rs_list=[]
            for i in range(len(edg_list)):
                conc=torch.cat((p_hid,out_list[i]),dim=1)                                       #Nx2d
                rs=torch.matmul(conc,self.rel_bt)                                             #Nx1
                rs_list+=[rs]

            rs=torch.stack(rs_list)                                                         #Nxr
            beta=F.softmax(rs,dim=0)                                                        #Nxr
            res=0
            for i in range(len(edg_list)):
                res+=out_list[i]*beta[i]
        else:
            res=0
            for i in range(len(edg_list)):
                res+=out_list[i]
        final_weight=torch.sigmoid(self.skip)
        res = self.out_norm(F.gelu(res)* (final_weight) + p_hid* (1 - final_weight))

        return res


    def message(self,edge_index,x_i, x_j,edge_type):
        
        node_f = torch.cat((x_i, x_j), 1)                                       #nx2d

        temp = torch.matmul(node_f, self.rel_wi[edge_type]).to(x_i.device)      #nx1

        alpha=softmax(temp,edge_index[1])

        rs=x_j*alpha                                                            #nxd
        return rs




class HRGATConv(nn.Module):
    def __init__(self,in_hid,out_hid,num_m1,num_m2,conv_name="hrgat",n_heads=8,n_layers=2,dropout=0.2,norm=True,hgt_layer=2,**kwargs):
        super(HRGATConv,self).__init__()
        self.conv_name=conv_name
        self.hetgat=nn.ModuleList()
        self.layer=n_layers

        self.hgt=nn.ModuleList()
        self.norm=nn.LayerNorm(out_hid)
        self.drop=Dropout(dropout)
        self.proj_a=nn.Linear(in_hid,out_hid,bias=False)
        self.proj_p=nn.Linear(in_hid,out_hid,bias=False)

        for _ in range(hgt_layer):
            if _ == 0:
                if self.conv_name == "rgsn":
                    self.hgt.append(RGSNConv(in_hid, out_hid, 1, num_m1))
                    self.hgt.append(RGSNConv(in_hid, out_hid, 1, num_m2))
                elif self.conv_name == "dgat":
                    self.hgt.append(DGATConv(in_hid, out_hid, num_m1,heads=n_heads))
                    self.hgt.append(DGATConv(in_hid, out_hid, num_m2,heads=n_heads))
            else:
                if self.conv_name == "rgsn":
                    self.hgt.append(RGSNConv(in_hid, out_hid, 1, num_m1))
                    self.hgt.append(RGSNConv(in_hid, out_hid, 1, num_m2))
                elif self.conv_name == "dgat":
                    self.hgt.append(DGATConv(out_hid, out_hid, num_m1,heads=n_heads))
                    self.hgt.append(DGATConv(out_hid, out_hid, num_m2,heads=n_heads))

        if self.conv_name == "dhan1":
            for n in range(n_layers):
                self.hetgat.append(HetGATConv(out_hid, out_hid,dual=False))
                self.hetgat.append(HetGATConv(out_hid, out_hid,dual=False))
        else:
            for n in range(n_layers):
                self.hetgat.append(HetGATConv(out_hid, out_hid))
                self.hetgat.append(HetGATConv(out_hid, out_hid))
    # _gh=(node_feature, edge_index, edge_type, id_list)
    def forward(self,paper_gh,author_gh,sub_graph,device):

        h_p=paper_gh[0].to(device)
        h_a=author_gh[0].to(device)
        for hl in range(int((len(self.hgt)/2))):
            if hl==0:
                h_p=self.hgt[2*hl](h_p,paper_gh[1].to(device),paper_gh[2].to(device))
                h_a=self.hgt[2*hl+1](h_a,author_gh[1].to(device),author_gh[2].to(device))
            else:
                h_p = self.hgt[2 * hl](h_p.to(device), paper_gh[1].to(device), paper_gh[2].to(device))
                h_a = self.hgt[2 * hl + 1](h_a.to(device), author_gh[1].to(device), author_gh[2].to(device))

        idl_a=author_gh[3]
        idl_p=paper_gh[3]
        edge_indx, edge_type=rel_graph(idl_a,idl_p,sub_graph)

        for ly in range(int(len(self.hetgat)/2)):
            p_hid = self.hetgat[2*ly](h_a, h_p, edge_indx.to(device), edge_type.to(device))

            edge_indx=torch.stack((edge_indx[1],edge_indx[0]))
            a_hid = self.hetgat[2*ly+1](h_p, h_a, edge_indx.to(device), edge_type.to(device))

            edge_indx = torch.stack((edge_indx[1], edge_indx[0]))
            h_a=a_hid
            h_p=p_hid
        h_a=self.drop(self.norm(h_a))
        h_p=self.drop(self.norm(h_p))


        return h_a,idl_a,h_p,idl_p