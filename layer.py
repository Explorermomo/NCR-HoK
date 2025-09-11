import datetime
import math
import numpy as np
import torch
import numpy
import random
from torch import nn
from torch.nn import Module, Parameter
from torch_geometric.nn import Linear, GCNConv, GATConv
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse

class GAT(nn.Module):
    def __init__(self, in_feature, out_feature, dropout, alpha=0.5, concat=True):
        super(GAT, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.FloatTensor(self.in_feature,self.out_feature))
        self.a = nn.Parameter(torch.FloatTensor(2*out_feature,1))
        self.reset_parameters()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_feature)
        nn.init.uniform_(self.W.data, -stdv, stdv)
        nn.init.uniform_(self.a.data, -stdv, stdv)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.shape[0]

        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1)\
            .view(N, -1, 2*self.out_feature)

        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(2)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT_nn(nn.Module):
    def __init__(self, in_feature, hidden_size, out_feature, dropout, num_heads=4):
        super(GAT_nn, self).__init__()
        self.gat_conv1 = GATConv(in_feature, hidden_size, concat=False, heads=1)
        self.gat_conv2 = GATConv(hidden_size, out_feature, concat=False,  heads=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x ,adj):
        edge_index, _ = dense_to_sparse(adj)
        x = self.gat_conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat_conv2(x, edge_index)
        return x

class HyperGAT(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, transfer=False, concat=False, bias=False):
        super(HyperGAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor( self.out_features, self.out_features))
        self.weight3_1 = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        self.weight3_1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj, Edge=None):  # x[1000, 128] adj:[1000,3000]


        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias      # x [1000, 128]

        x_4att = x.matmul(self.weight2)          # [1000,128] * [128, 64] = [1000, 64]

        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[2]  # number of node

        pair = adj.nonzero().t()

        if Edge is not None:
            edge = Edge
            edge_4att = edge.matmul(self.weight3_1)  # [3000, 500] * [500, 30] = [3000, 50]
        else:
            get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
            x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

            q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0], 1).view(x1.shape[0],
                                                                                      self.out_features)

            pair_h = torch.cat((q1, x1), dim=-1)
            pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
            assert not torch.isnan(pair_e).any()
            pair_e = F.dropout(pair_e, self.dropout, training=self.training)

            e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()  # [1, 3000, 1000]

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)  # [1, 1000, 1000]

            attention_edge = F.softmax(attention, dim=2)

            # edge = torch.matmul(attention_edge, x.squeeze())  # [1, 3000, 1000] * [1000, 1000]
            edge = torch.matmul(attention_edge, x_4att)

            edge = F.dropout(edge, self.dropout, training=self.training)

            edge_4att = edge.matmul(self.weight3) # [3000,1000] * [1000, 500] = [3000,500]

        get = lambda i: edge_4att[i][adj[i].nonzero().t()[0]]
        y1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        q1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        pair_h = torch.cat((q1, y1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()   # [1,3000,1000]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1, 2), dim=2)

        node = torch.matmul(attention_node, edge_4att)  # [1,1000,3000] * [ 3000, 500] = [1, 1000, 500]

        if self.concat:
            node = F.elu(node)

        return node, edge

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DegreeCentralityEncoder(nn.Module):
    def __init__(self,  embedding_size, node_num=1000):
        super(DegreeCentralityEncoder, self).__init__()
        self.x_minus = nn.Embedding(node_num, embedding_size)
        self.x_plus = nn.Embedding(node_num, embedding_size)
        self.BC = nn.Embedding(node_num, embedding_size)

    def forward(self, adj, undirected = False, Betweeness_C = None):
        in_degrees = torch.sum(adj, dim=0).long()
        out_degrees = torch.sum(adj, dim=1).long()
        node_initial_feature = torch.cat(
                (self.x_minus(in_degrees), self.x_plus(out_degrees)), dim=1)
        if Betweeness_C is not None:
            node_initial_feature = torch.cat(
                (node_initial_feature, self.BC(Betweeness_C)), dim=1)

        return node_initial_feature

class Fiter(nn.Module):
    def __init__(self, output_line_length):
        super(Fiter, self).__init__()
        self.bias = nn.Parameter(torch.randn(1,output_line_length))

    def reset_parameters(self):
        self.bias.data.uniform_(0, 1)

    def forward(self, pre_line):
        x = pre_line + self.bias

        return x


class MLPdecoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, output_size=10):
        super(MLPdecoder, self).__init__()
        self.fc  = nn.Linear(input_size, hidden_size)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.ReLU(x)
        x = self.fc2(x)

        return x





























