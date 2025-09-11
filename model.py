import datetime
import math
import numpy as np
import torch
import random
import os
import scipy.io as sio
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
import time
import glob
import os

from torch import nn
from torch.nn import Module, Parameter
from torch_geometric.nn import Linear
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

from layer import *
from utilzes import *
from node_attack import *


def random_delete_nodes(adj_matrix, target_nodes=1000):
    current_nodes = adj_matrix.shape[0]
    if current_nodes <= target_nodes:
        return adj_matrix 
    random_indices = np.random.choice(current_nodes, target_nodes, replace=False)
    new_adj_matrix = adj_matrix[random_indices][:, random_indices]

    return new_adj_matrix

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class Loss_function(nn.Module):
    def __init__(self):
        super(Loss_function, self).__init__()
    def forward(self, pred_line, ground_ture):
        # x = torch.sum(torch.pow(pred_line - ground_ture, 2), dim=1)
        loss = torch.mean(torch.pow(pred_line - ground_ture, 2), dim=1)
        return loss

def load_batch(mat, idx, batch_size, begin_index):
    A = []
    y = []
    for i in range(batch_size):
        tmpA = mat['data_train'][idx[i+begin_index], 0]['A'][0, 0].todense()
        tmpy = np.squeeze(mat['data_train'][idx[i+begin_index], 0]['y'][0, 0])
        A.append(np.expand_dims(tmpA, axis=0))
        # A.append(tmpA)
        y.append(tmpy)
    return A, y


# load cross validation data
def load_batch_val(mat, batch_size, begin_index):
    A = []
    y = []
    for i in range(batch_size):
        tmpA = mat['data_x'][i + begin_index, 0]['A'][0, 0].todense()
        tmpy = np.squeeze(mat['data_x'][i + begin_index, 0]['y'][0, 0])
        A.append(np.expand_dims(tmpA, axis=2))
        y.append(tmpy)
    return A, y

class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.5):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGAT(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
                                                   concat=False)
        self.gat2 = HyperGAT(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=False,
                                                   concat=False)

    def forward(self, x, H):
        x1, xe1 = self.gat1(x, H,  Edge = None)
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x3, xe = self.gat2(x2, H, Edge=xe1)

        return x3, xe

### Module_define
class Robustness_predict_modul(Module):
    def __init__(self,opt, graph_number, line_len):
        super(Robustness_predict_modul, self).__init__()
        self.dropout = opt.dropout
        self.in_feature = opt.initialFeatureSize
        self.hidden_size = opt.n_hid
        self.out_feature = opt.outputFeatureSize
        self.End_feature = opt.EndNodeFeature
        self.degree_size = opt.degree_size
        self.clip_value = opt.clip_value


        ### GAT Backbone: output is [N, out_feature]
        # self.GatBone1 = GAT(self.in_feature, self.hidden_size, self.dropout)
        # self.GatBone2 = GAT(self.hidden_size, self.out_feature, self.dropout)

        self.GatBone = GAT_nn(self.in_feature, self.hidden_size, self.out_feature, self.dropout)

        # self.GATFC = nn.Linear(self.out_feature, self.out_feature)
        # self.GATDropout = nn.Dropout(self.dropout)

        self.Degree_encoder = DegreeCentralityEncoder(self.degree_size)

        ### hypergraph network: output is [N, out_feature], then combine the row adj, inputing the hypergraph generate methods to generate the new hypergraph stucture.
        ###                     (the node feature in k layer is k-1 layer's output, when k=0, the node feature is adj.)
        ###                     (the last output is [N, out_feature].)
        # self.hgnn_one1 = HyperGAT(self.out_feature, 5, dropout=self.dropout, alpha=0.2, transfer=False, concat=False)  # one layer
        # self.hgnn_one2 = HyperGAT(5, 1, dropout=self.dropout, alpha=0.2, transfer=False,
        #                          concat=False)

        self.hgnn2 = HGNN_ATT(self.in_feature, self.hidden_size, self.out_feature, dropout=self.dropout)  # two layer
        self.hgnn3 = HGNN_ATT(self.out_feature, 5, 1, dropout=self.dropout)  # two layer


        ### Hypergraph Non-linear output layer
        # self.first_fc = nn.Linear(self.hidden_size, self.hidden_size)
        # self.first_fc_dropout = nn.Dropout(self.dropout)
        # self.second_fc = nn.Linear(self.out_feature, self.out_feature)
        # self.second_fc_dropout = nn.Dropout(self.dropout)

        # Linear
        self.Linear = nn.Linear(self.out_feature, 1 , bias=True)
        self.Linear2 = nn.Linear(self.out_feature, self.End_feature, bias=True)
        # self.Linear3 = nn.Linear(self.End_feature, 1)
        # self.BCLinear = nn.Linear( 1, self.degree_size)

        self.BCDecoder = MLPdecoder()
        # self.InDegreeDecoder = MLPdecoder()
        # self.OutDegreeDecoder = MLPdecoder()

        ### the end MLP structure
        l_hid = int(graph_number*(2*self.End_feature))
        # l_hid2 = int(l_hid / 2)
        # self.layer_normH = nn.LayerNorm(self.End_feature*1000*3, eps=1e-6)
        self.fc = nn.Linear(graph_number*(2*self.End_feature), l_hid, bias=True)
        # self.fc2 = nn.Linear(l_hid,l_hid2)
        self.fc3 = nn.Linear(l_hid, line_len, bias=True)
        self.fc_dropout = nn.Dropout(self.dropout)
        # self.fc2_dropout = nn.Dropout(self.dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2) 
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.Loss = nn.SmoothL1Loss() 

    def forward(self, A, hypergraph_adj, adj, hypergraph_khop_and_k_shell, batch_size = 5, BC = None):

        BC_f = self.BCDecoder(BC.T)   # 1 20 10
        #
        x = self.Degree_encoder(adj)  #  1 10 20

        x = torch.cat((x, BC_f), dim=1)  # 30

        output_feature_of_GAT = self.GatBone(x, adj)  # 30 60 10
        output_feature_of_GAT = self.Linear(output_feature_of_GAT)  # 10 2

        x = torch.unsqueeze(x, dim=0)
        hypergraph_adj = torch.unsqueeze(hypergraph_adj.T, dim=0)
        hypergraph_embedding, _ = self.hgnn2(x, hypergraph_adj)  # 30 60 10

        hypergraph_embedding = torch.squeeze(hypergraph_embedding, dim=0)
        hypergraph_embedding0 = self.Linear2(hypergraph_embedding)  # 10 2


        hypergraph_embedding1 = hypergraph_embedding.detach()
        hypergraph_knn = construct_H_with_KNN(hypergraph_embedding1).cuda()

        hypergraph_embedding = torch.unsqueeze(hypergraph_embedding, dim=0)
        hypergraph = torch.unsqueeze(hypergraph_knn.T, dim=0)

        ### use hgnn3
        hypergraph_embedding, _ = self.hgnn3(hypergraph_embedding, hypergraph)  # 10 20 2
        hypergraph_embedding = torch.squeeze(hypergraph_embedding, dim=0)

        embedding = torch.concat((output_feature_of_GAT, hypergraph_embedding0, hypergraph_embedding), dim=1)  # 2 2 2 6
        #
        # ### MLP output the controlibility robustness curve
        x = embedding.reshape(1,-1)
        x = F.leaky_relu(self.fc(x))
        x = self.fc_dropout(x)
        x = F.sigmoid(self.fc3(x))

        return x

    # having running
    def TrainM(self, mat,epoch=None, begin_time=None, all_train_index = None, khop=2, knn=True, kshell=True,kmeans=True, batch_size = 1):
        self.train()
        self.scheduler.step()
        num_train_ins = len(mat['data_train'])
        shuffle_list_train = random.sample(range(0, num_train_ins), num_train_ins) 


        save_fold1, save_fold2, save_fold3, save_fold4 = '/home/wu/The_second_project/PCR(GAT)/Hypergraph_adj_mateix_degree2',\
                                             '/home/wu/The_second_project/PCR(GAT)/Hypergraph_adj_mateix_degree3', \
                                             '/home/wu/The_second_project/PCR(GAT)/Betweeness_Centrality',\
                                             '/home/wu/The_second_project/PCR(GAT)/K-shell'

        loss_value = []
        eachtime = []
        num_batches = (num_train_ins + batch_size - 1) // batch_size  
        total_loss=0
        pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}' if epoch is not None else 'Training')
        for i in pbar:
        # for sort, i in enumerate(all_train_index):
            A = np.array(mat['data_train'][shuffle_list_train[i], 0]['A'][0, 0].todense())  
            y = np.squeeze(mat['data_train'][shuffle_list_train[i], 0]['y'][0, 0])
            # A = np.array(mat['data_train'][all_train_index[shuffle_list_train[i]], 0]['A'][0, 0].todense())  
            # y = np.squeeze(mat['data_train'][all_train_index[shuffle_list_train[i]], 0]['y'][0, 0])

            save_path2 = os.path.join(save_fold2, f'hypergraph_3degree_{shuffle_list_train[i]}.mat')
            save_path3 = os.path.join(save_fold3, f'graph_BC_{shuffle_list_train[i]}.mat')
            # save_path2 = os.path.join(save_fold2, f'hypergraph_3degree_{all_train_index[shuffle_list_train[i]]}.mat')
            # save_path3 = os.path.join(save_fold3, f'graph_BC_{all_train_index[shuffle_list_train[i]]}.mat')

            hypergraph_khop_dict = scipy.io.loadmat(save_path2)
            BC_dict = scipy.io.loadmat(save_path3)
            # k_shell_g = scipy.io.loadmat(save_path4)
            hypergraph_khop = hypergraph_khop_dict['A']
            # hypergraph_kshell = k_shell_g['A']
            BC = BC_dict['BC']

            ### hypegraph generate and embedding
            # hypergraph_khop = generate_khop_hypergraph(A, 3)
            # hypergraph_knn = construct_H_with_KNN(A)
            # hypergraph_kshell = generate_kshell_hypergraph(A)
            # hypergraph_kmeans = construct_edge_list_from_Kmeans(A, 5, 1, 20)
            ## generate the whole generation
            # hypergraph_khop_and_k_shell = np.concatenate((hypergraph_khop,hypergraph_kshell), axis=1)
            # hypergraph = torch.from_numpy(np.concatenate((hypergraph_khop, hypergraph_kshell), axis=1)).cuda()
            # hypergraph = torch.from_numpy(hypergraph_khop.todense()).cuda()  # khop=2\khop=4
            hypergraph = torch.from_numpy(hypergraph_khop).cuda()
            BC = torch.FloatTensor(BC).cuda()
            A = torch.FloatTensor(A).cuda()
            Y = torch.FloatTensor(y).cuda()


            output = self.forward(A, hypergraph, A, hypergraph, BC=BC)

            loss = self.Loss(output, Y)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
            self.optimizer.step()
            total_loss += loss.item()
        
            avg_loss_so_far = total_loss / (i + 1)
            pbar.set_postfix({'Loss': f'{avg_loss_so_far:.4f}'})

        
            each_graph_time = time.time()
            eachtime.append(each_graph_time - begin_time)
            loss_value.append(loss.cpu().detach().numpy())
            
            # print(f'Training  Loss{sort}={loss}')
        avg_loss = total_loss / num_train_ins
        return loss_value, eachtime,avg_loss

    def TrainM_extension_nodenumber(self, Graph_node_number=800):
        self.train()
        self.scheduler.step()
        num_train_ins = 800*4
        shuffle_list_train = random.sample(range(1, num_train_ins+1), num_train_ins)
        total_loss = 0
        save_fold1, save_fold2 = \
            f'/home/wu/Controlbility_robustness_predict/train80010001200/Hypergraph_adj_{Graph_node_number}', \
            f'/home/wu/Controlbility_robustness_predict/train80010001200/BC_{Graph_node_number}'
        pbar = tqdm(range(num_train_ins), desc=f'Epoch {epoch+1}' if epoch is not None else 'Training')
        for i in pbar:
            save_fold = f'/home/wu/Controlbility_robustness_predict/data_{Graph_node_number}_train/RA_10_*_{Graph_node_number}_{shuffle_list_train[i]}.mat'
            path = glob.glob(save_fold)
            A = scipy.io.loadmat(path[0])[f'Graph_{shuffle_list_train[i]}']['A'][0, 0].todense()
            target = np.squeeze(scipy.io.loadmat(path[0])[f'Graph_{shuffle_list_train[i]}']['Y'][0, 0])[:-1]

            save_path1 = os.path.join(save_fold1, f't_hypergraph_3degree_*_{shuffle_list_train[i]}.mat')
            save_path2 = os.path.join(save_fold2, f't_graph_BC_*_{shuffle_list_train[i]}.mat')
            path1 = glob.glob(save_path1)
            path2 = glob.glob(save_path2)
            hypergraph_khop_dict = scipy.io.loadmat(path1[0])
            BC_dict = scipy.io.loadmat(path2[0])
            hypergraph_khop = hypergraph_khop_dict['A'].todense()
            BC = BC_dict['BC']

            hypergraph = torch.from_numpy(hypergraph_khop).cuda()
            BC = torch.FloatTensor(BC).cuda()
            A = torch.FloatTensor(A).cuda()
            Y = torch.FloatTensor(target).cuda()

            output = self.forward(A, hypergraph, A, hypergraph, BC=BC)

            loss = self.Loss(output, Y)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
            self.optimizer.step()

            
         
            total_loss += loss.item()
        
            avg_loss_so_far = total_loss / (i + 1)
            pbar.set_postfix({'Loss': f'{avg_loss_so_far:.4f}'})

        avg_loss = total_loss / num_train_ins
    # having running:  (TrainM:TestM)
    def TestM(self, mat,save_result, save_ground_true_line, k_degree = 0, type=0,  khop=2, knn=True, kshell=True, batch_size = 1, vison=0, kmeans=True, EndEpoch = None, Exp_result_error = None, Exp_result_sd = None):
        self.eval()
        self.optimizer.zero_grad()
        if vison == 1 and EndEpoch == 9:
            fig, axs = plt.subplots(4, 4, figsize=(20, 20))

        GraphTYPE = ['ER', 'SF', 'QSN', 'SW']
        GraphKEGREE = [2, 5, 8, 10]

        with torch.no_grad():
            # result, ground_true_line = [], []
            sub_result = np.zeros((4, 4))
            # for type in range(4):
            outer_pbar = tqdm(total=16, desc="Testing Progress")  # 4x4=16
            for type, type_name in enumerate(['ER','QSN','SF', 'SW']):
            # for type in ['ER','QSN','SF', 'SW']:
            #     for k_degree in range(4):
                for k_degree, k_number in enumerate([2,5,8,10]):
                # for type in ['ER', 'QSN', 'SF', 'SW']:
                    save_fold1, save_fold2, save_fold3, save_fold4 \
                        = f'/home/wu/The_second_project/PCR(GAT)/test_data/Hypergraph_adj_mateix_degree2_{type}_{k_degree}', \
                          f'/home/wu/The_second_project/PCR(GAT)/test_data/Hypergraph_adj_mateix_degree3_{type}_{k_degree}', \
                          f'/home/wu/The_second_project/PCR(GAT)/test_data/Betweeness_Centrality_{type}_{k_degree}', \
                          f'/home/wu/The_second_project/PCR(GAT)/test_data/K-shell_{type}_{k_degree}'
                    result, ground_true_line = [], []
                    inner_pbar = tqdm(range(100), desc=f'Type{type}-Deg{k_degree}', leave=False)
                    for i in inner_pbar:
                        A = mat['data_test'][type, k_degree, i]['A'][0, 0].todense()
                        targer = np.squeeze(mat['data_test'][type, k_degree, i]['y'][0, 0])
                        # A = scipy.io.loadmat(f'/home/wu/Controlbility_robustness_predict/data_800_val/RA_10_{type}_800_{i+1}')[f'Graph_{i+1}']['A'][0, 0].todense()
                        # targer = np.squeeze(scipy.io.loadmat(f'/home/wu/Controlbility_robustness_predict/data_800_val/RA_10_{type}_800_{i+1}')[f'Graph_{i+1}']['Y'][0, 0])[:-1]

                        # save_path1 = os.path.join(save_fold1, f't_hypergraph_2degree_{i}.mat')
                        save_path2 = os.path.join(save_fold2, f't_hypergraph_3degree_{i}.mat')
                        save_path3 = os.path.join(save_fold3, f't_graph_BC_{i}.mat')
                        # save_path4 = os.path.join(save_fold4, f't_graph_K_shell_{i}.mat')
                        hypergraph_khop_dict = scipy.io.loadmat(save_path2)
                        BC_dict = scipy.io.loadmat(save_path3)
                        # hypergraph_kshell_dict = scipy.io.loadmat(save_path4)
                        hypergraph_khop = hypergraph_khop_dict['A']
                        # hypergraph_kshell = hypergraph_kshell_dict['A']
                        BC = BC_dict['BC']

                        # hypergraph = torch.from_numpy(np.concatenate((hypergraph_khop, hypergraph_kshell), axis=1)).cuda()
                        # hypergraph = torch.from_numpy(hypergraph_khop.todense()).cuda() # khop=2\khop=4
                        hypergraph = torch.from_numpy(hypergraph_khop).cuda()
                        BC = torch.FloatTensor(BC).cuda() 
                        A = torch.FloatTensor(A).cuda()


                        output = self.forward(A, hypergraph, A, hypergraph,  BC = BC)

                        # output = self.forward(A,A)
                        outputs = trans_to_cpu(torch.squeeze(output)).numpy()

                        error = np.mean(np.abs(outputs - targer))
                        inner_pbar.set_postfix({'Error': f'{error:.4f}'})


                        result.append(outputs)
                        ground_true_line.append(targer)
                    inner_pbar.close()
                    result = np.array(result)  # 100 999
                    ground_true_line = np.array(ground_true_line)

                    if vison == 1 and EndEpoch == 9:
                        pv = np.mean(result, axis=0)
                        tv = np.mean(ground_true_line, axis=0)
                        er = np.mean(np.abs(result - ground_true_line), axis=0)
                        st = np.std(np.abs(result - ground_true_line), axis=0)
                        x = np.arange(len(pv))
                        axs[type, k_degree].plot(x, tv, color='red', linewidth=2.5)
                        axs[type, k_degree].plot(x, pv, color='blue', linestyle='--', linewidth=2.5)
                        axs[type, k_degree].plot(x, er, color='black', linewidth=2.5)
                        axs[type, k_degree].plot(x, st, color='green', linestyle='--', linewidth=2.5)
                        axs[type, k_degree].set_title(f'RA {GraphTYPE[type]}-<K>={GraphKEGREE[k_degree]}', fontsize=16)
                        axs[type, k_degree].set_xlabel('P$_N$', fontsize=12)
                        axs[type, k_degree].set_ylabel('n$_D$', fontsize=12)
                        axs[type, k_degree].set_yscale('log')
                        axs[type, k_degree].grid(True)
                        # axs.grid(True)

                    # x = np.mean(np.abs(result-ground_true_line), axis=1)
                    average_std = np.mean(np.std(np.abs(result-ground_true_line), axis=0))
                    average_error = np.mean(np.mean(np.abs(result-ground_true_line), axis=1))
                    sub_result[type, k_degree] = average_error
                    if Exp_result_error[type, k_degree] > average_error:
                        Exp_result_error[type, k_degree] = average_error
                        Exp_result_sd[type, k_degree] = average_std

                        save_ground_true_line[f'{type_name}_{k_number}'] = ground_true_line
                        save_result[f'{type_name}_{k_number}'] = result
                    outer_pbar.update(1)
                    outer_pbar.set_postfix({
                        'Current Error': f'{average_error:.4f}',
                        'Best Error': f'{np.min(Exp_result_error):.4f}'
                    })
        outer_pbar.close()
        print("Test Results Matrix:")
            
        print(sub_result)
        if vison == 1 and EndEpoch == 9:
            plt.tight_layout()
            plt.savefig('/home/wu/The_second_project/PCR(GAT)/Split80Exp.png')

        save_result_fold = '/home/wu/The_second_project/PCR(GAT)/result'
        save_path1 = os.path.join(save_result_fold, f'TestM_khop_4_ground_true_line.mat')
        save_path2 = os.path.join(save_result_fold, f'TestM_khop_4_predicting_line.mat')
        scipy.io.savemat(save_path1, save_ground_true_line)
        scipy.io.savemat(save_path2, save_result)

        return Exp_result_error, Exp_result_sd, sub_result, save_ground_true_line, save_result

    def TestM_extension_kdgree(self, mat, save_result,save_ground_true_line,k_degree = 0, type=0,  khop=2, knn=True, kshell=True, batch_size = 1, vison=0, kmeans=True, EndEpoch = None, Exp_result_error = None, Exp_result_sd = None):
        self.eval()
        self.optimizer.zero_grad()
        if vison == 1 and EndEpoch == 9:
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))

        with torch.no_grad():
            sub_result = np.zeros((2, 4))
            time_graph = []
            for k_sort, k_degree in enumerate(['4', '7']):
                for type_sort, type in enumerate(['ER', 'SF', 'QSN', 'SW']):
                    save_fold1, save_fold2 = \
                        f'/home/wu/The_second_project/PCR(GAT)/Extension of experiments/k4_k7/Hypergraph_adj_mateix_degree3_{k_degree}', \
                        f'/home/wu/The_second_project/PCR(GAT)/Extension of experiments/k4_k7/BC_{k_degree}'
                    result, ground_true_line = [], []
                    for i in range(100):
                        save_fold = f'/home/wu/Controlbility_robustness_predict/data_k{k_degree}/RA_{k_degree}_{type}_1000_{i + 1+type_sort*100}.mat'
                        A = scipy.io.loadmat(save_fold)[f'Graph_{i+1+type_sort*100}']['A'][0, 0].todense()
                        targer = np.squeeze(scipy.io.loadmat(save_fold)[f'Graph_{i+1+type_sort*100}']['Y'][0, 0])[:-1]

                        save_path1 = os.path.join(save_fold1, f't_hypergraph_3degree_{type}_{i+ 1 + type_sort * 100}.mat')
                        save_path2 = os.path.join(save_fold2, f't_graph_BC_{type}_{i+ 1 + type_sort * 100}.mat')
                        hypergraph_khop_dict = scipy.io.loadmat(save_path1)
                        BC_dict = scipy.io.loadmat(save_path2)
                        hypergraph_khop = hypergraph_khop_dict['A'].todense()
                        BC = BC_dict['BC']

                        hypergraph = torch.from_numpy(hypergraph_khop).cuda()
                        # BC = compute_BC(A)
                        BC = torch.FloatTensor(BC).cuda()
                        # BC = torch.unsqueeze(BC, dim=0)
                        A = torch.FloatTensor(A).cuda()


                        start_time = time.time()
                        output = self.forward(A, hypergraph, A, hypergraph,  BC = BC)
                        end_time = time.time()
                        time_graph.append(end_time-start_time)

                        outputs = trans_to_cpu(torch.squeeze(output)).numpy()



                        print(f'Graph Type{type}, Node ave_degrees{k_degree}, Error value{i}={np.mean(np.abs(outputs-targer))}', end='\n')

                        result.append(outputs)
                        ground_true_line.append(targer)


                    result = np.array(result)  # 100 999
                    ground_true_line = np.array(ground_true_line)


                    if vison == 1 and EndEpoch == 9:
                        pv = np.mean(result, axis=0)
                        tv = np.mean(ground_true_line, axis=0)
                        er = np.mean(np.abs(result - ground_true_line), axis=0)
                        st = np.std(np.abs(result - ground_true_line), axis=0)
                        x = np.arange(len(pv))
                        axs[k_sort, type_sort].plot(x, tv, color='red', linewidth=2.5)
                        axs[k_sort, type_sort].plot(x, pv, color='blue', linestyle='--', linewidth=2.5)
                        axs[k_sort, type_sort].plot(x, er, color='black', linewidth=2.5)
                        axs[k_sort, type_sort].plot(x, st, color='green', linestyle='--', linewidth=2.5)
                        axs[k_sort, type_sort].set_title(f'RA {type}-<K>={k_degree}', fontsize=16)
                        axs[k_sort, type_sort].set_xlabel('P$_N$', fontsize=12)
                        axs[k_sort, type_sort].set_ylabel('n$_D$', fontsize=12)
                        axs[k_sort, type_sort].set_yscale('log')
                        axs[k_sort, type_sort].grid(True)


                    # x = np.mean(np.abs(result-ground_true_line), axis=1)
                    average_std = np.mean(np.std(np.abs(result-ground_true_line), axis=0))
                    average_error = np.mean(np.mean(np.abs(result-ground_true_line), axis=1))
                    sub_result[k_sort, type_sort] = average_error
                    if Exp_result_error[k_sort, type_sort] > average_error:
                        Exp_result_error[k_sort, type_sort] = average_error
                        Exp_result_sd[k_sort, type_sort] = average_std

                        save_ground_true_line[f'{type}_{k_degree}'] = ground_true_line
                        save_result[f'{type}_{k_degree}'] = result

            print(np.mean(np.array(time_graph)))

        if vison == 1 and EndEpoch == 9:

            plt.tight_layout()
            # plt.show()
            plt.savefig('/home/wu/The_second_project/PCR(GAT)/SecondEXP_k.png')
        return Exp_result_error, Exp_result_sd, sub_result, save_ground_true_line, save_result

    # having running: graph nodenumber : TrainM_extension_nodenumber:TestM_extension_nodenumber 
    def TestM_extension_nodenumber(self, save_ground_true_line, save_result, vison=0, Graph_node_number=800, EndEpoch = None, Exp_result_error = None, Exp_result_sd = None):
        self.eval()
        self.optimizer.zero_grad()
        if vison == 1 and EndEpoch == 9:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        with torch.no_grad():
            sub_result = np.zeros((1, 4))
            time_graph = []
            for type_sort, type in enumerate(['ER', 'SF', 'QSN', 'SW']):
                save_fold1, save_fold2 = \
                    f'/home/wu/Controlbility_robustness_predict/val80010001200/Hypergraph_adj_{Graph_node_number}', \
                    f'/home/wu/Controlbility_robustness_predict/val80010001200/BC_{Graph_node_number}'
                result, ground_true_line = [], []
                for i in range(100):
                    save_fold = f'/home/wu/Controlbility_robustness_predict/data_{Graph_node_number}_val/RA_10_{type}_{Graph_node_number}_{i + 1+type_sort*100}.mat'
                    A = scipy.io.loadmat(save_fold)[f'Graph_{i+1+type_sort*100}']['A'][0, 0].todense()
                    targer = np.squeeze(scipy.io.loadmat(save_fold)[f'Graph_{i+1+type_sort*100}']['Y'][0, 0])[:-1]

                    save_path1 = os.path.join(save_fold1, f't_hypergraph_3degree_{type}_{i+ 1 + type_sort * 100}.mat')
                    save_path2 = os.path.join(save_fold2, f't_graph_BC_{type}_{i+ 1 + type_sort * 100}.mat')
                    hypergraph_khop_dict = scipy.io.loadmat(save_path1)
                    BC_dict = scipy.io.loadmat(save_path2)
                    hypergraph_khop = hypergraph_khop_dict['A'].todense()
                    BC = BC_dict['BC']

                    hypergraph = torch.from_numpy(hypergraph_khop).cuda()
                    BC = torch.FloatTensor(BC).cuda()
                    A = torch.FloatTensor(A).cuda()

                    start_time = time.time()
                    output = self.forward(A, hypergraph, A, hypergraph,  BC = BC)
                    end_time = time.time()
                    time_graph.append(end_time-start_time)

                    outputs = trans_to_cpu(torch.squeeze(output)).numpy()


                    print(f'Graph Type{type}, Node number{Graph_node_number}, Error value{i}={np.mean(np.abs(outputs-targer))}', end='\n')


                    result.append(outputs)
                    ground_true_line.append(targer)


                result = np.array(result)
                ground_true_line = np.array(ground_true_line)


                if vison == 1 and EndEpoch == 9:
                    pv = np.mean(result, axis=0)
                    tv = np.mean(ground_true_line, axis=0)
                    er = np.mean(np.abs(result - ground_true_line), axis=0)
                    st = np.std(np.abs(result - ground_true_line), axis=0)
                    x = np.arange(len(pv))
                    axs[type_sort].plot(x, tv, color='red', linewidth=2.5)
                    axs[type_sort].plot(x, pv, color='blue', linestyle='--', linewidth=2.5)
                    axs[type_sort].plot(x, er, color='black', linewidth=2.5)
                    axs[type_sort].plot(x, st, color='green', linestyle='--', linewidth=2.5)
                    axs[type_sort].set_title(f'RA {type}-<K>=10', fontsize=16)
                    axs[type_sort].set_xlabel('P$_N$', fontsize=12)
                    axs[type_sort].set_ylabel('n$_D$', fontsize=12)
                    axs[type_sort].set_yscale("log")
                    axs[type_sort].grid(True)


                # x = np.mean(np.abs(result-ground_true_line), axis=1)
                average_std = np.mean(np.std(np.abs(result-ground_true_line), axis=0))
                average_error = np.mean(np.mean(np.abs(result-ground_true_line), axis=1))
                sub_result[0, type_sort] = average_error
                if Exp_result_error[0, type_sort] > average_error:
                    Exp_result_error[0, type_sort] = average_error
                    Exp_result_sd[0, type_sort] = average_std

                    save_ground_true_line[f'{type}_{Graph_node_number}'] = ground_true_line
                    save_result[f'{type}_{Graph_node_number}'] = result

        print(np.mean(np.array(time_graph)))

        if vison == 1 and EndEpoch == 9:
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'/home/wu/The_second_project/PCR(GAT)/graph_node_number_{Graph_node_number}.png')
        return Exp_result_error, Exp_result_sd, sub_result, save_ground_true_line, save_result

    # having running: training split <k>=8 : TrainM:TestM_extension_split_k8
    def TestM_extension_split_k8(self, save_result, save_ground_true_line , vison=0, EndEpoch = None, Exp_result_error = None, Exp_result_sd = None):
        self.eval()
        self.optimizer.zero_grad()
        if vison == 1 and EndEpoch == 9:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        with torch.no_grad():
            sub_result = np.zeros((1, 4))
            time_graph = []
            for type_sort, type in enumerate(['ER', 'SF', 'QSN', 'SW']):
                save_fold1, save_fold2 = \
                    f'/home/wu/Controlbility_robustness_predict/val_1000_k8_AandBC/Hypergraph_adj_{type}', \
                    f'/home/wu/Controlbility_robustness_predict/val_1000_k8_AandBC/BC_{type}'
                result, ground_true_line = [], []
                for i in range(100):
                    save_fold = f'/home/wu/Controlbility_robustness_predict/data_1000_k8_val/RA_8_{type}_1000_{i + 1+type_sort*100}.mat'
                    A = scipy.io.loadmat(save_fold)[f'Graph_{i+1+type_sort*100}']['A'][0, 0].todense()
                    targer = np.squeeze(scipy.io.loadmat(save_fold)[f'Graph_{i+1+type_sort*100}']['Y'][0, 0])[:-1]

                    save_path1 = os.path.join(save_fold1, f't_hypergraph_3degree_{type}_{i+ 1 + type_sort * 100}.mat')
                    save_path2 = os.path.join(save_fold2, f't_graph_BC_{type}_{i+ 1 + type_sort * 100}.mat')
                    hypergraph_khop_dict = scipy.io.loadmat(save_path1)
                    BC_dict = scipy.io.loadmat(save_path2)
                    hypergraph_khop = hypergraph_khop_dict['A'].todense()
                    BC = BC_dict['BC']

                    hypergraph = torch.from_numpy(hypergraph_khop).cuda()
                    BC = torch.FloatTensor(BC).cuda()
                    A = torch.FloatTensor(A).cuda()

                    start_time = time.time()
                    output = self.forward(A, hypergraph, A, hypergraph,  BC = BC)
                    end_time = time.time()
                    time_graph.append(end_time-start_time)

                    outputs = trans_to_cpu(torch.squeeze(output)).numpy()


                    print(f'Graph Type{type}, Error value{i}={np.mean(np.abs(outputs-targer))}', end='\n')


                    result.append(outputs)
                    ground_true_line.append(targer)


                result = np.array(result)
                ground_true_line = np.array(ground_true_line)

                if vison == 1 and EndEpoch == 9:
                    pv = np.mean(result, axis=0)
                    tv = np.mean(ground_true_line, axis=0)
                    er = np.mean(np.abs(result - ground_true_line), axis=0)
                    st = np.std(np.abs(result - ground_true_line), axis=0)
                    x = np.arange(len(pv))
                    axs[type_sort].plot(x, tv, color='red', linewidth=2.5)
                    axs[type_sort].plot(x, pv, color='blue', linestyle='--', linewidth=2.5)
                    axs[type_sort].plot(x, er, color='black', linewidth=2.5)
                    axs[type_sort].plot(x, st, color='green', linestyle='--', linewidth=2.5)
                    axs[type_sort].set_title(f'RA {type}-<K>=10', fontsize=16)
                    axs[type_sort].set_xlabel('P$_N$', fontsize=12)
                    axs[type_sort].set_ylabel('n$_D$', fontsize=12)
                    axs[type_sort].set_yscale("log")
                    axs[type_sort].grid(True)


                # x = np.mean(np.abs(result-ground_true_line), axis=1)
                average_std = np.mean(np.std(np.abs(result-ground_true_line), axis=0))
                average_error = np.mean(np.mean(np.abs(result-ground_true_line), axis=1))
                sub_result[0, type_sort] = average_error
                if Exp_result_error[0, type_sort] > average_error:
                    Exp_result_error[0, type_sort] = average_error
                    Exp_result_sd[0, type_sort] = average_std

                    save_ground_true_line[f'{type_sort}'] = ground_true_line
                    save_result[f'{type_sort}'] = result

        save_result_fold = '/home/wu/The_second_project/PCR(GAT)/result'
        save_path1 = os.path.join(save_result_fold, f'TestM_training_20_ground_true_line.mat')
        save_path2 = os.path.join(save_result_fold, f'TestM_training_20_predicting_line.mat')
        scipy.io.savemat(save_path1, save_ground_true_line)
        scipy.io.savemat(save_path2, save_result)

        if vison == 1 and EndEpoch == 9:
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'/home/wu/The_second_project/PCR(GAT)/k8_Split80Exp.png')


        return Exp_result_error, Exp_result_sd, sub_result

    # the test of real world network
    def TestM_real_world_network(self, save_result, save_ground_true_line, k_degree = 0, type=0,  khop=2, knn=True, kshell=True, batch_size = 1, vison=0, kmeans=True, EndEpoch = None, Exp_result_error = None, Exp_result_sd = None):
        self.eval()
        self.optimizer.zero_grad()

        DDG = '/home/wu/The_second_project/PCR(GAT)/RealWorldNetwork/DD_g79.txt'
        DEL = '/home/wu/The_second_project/PCR(GAT)/RealWorldNetwork/delaunay_n10.txt'
        DWT5 = '/home/wu/The_second_project/PCR(GAT)/RealWorldNetwork/dwt_1005.txt'
        DWT7 = '/home/wu/The_second_project/PCR(GAT)/RealWorldNetwork/dwt_1007.txt'
        LSH = '/home/wu/The_second_project/PCR(GAT)/RealWorldNetwork/lshp1009.txt'
        ORS = '/home/wu/The_second_project/PCR(GAT)/RealWorldNetwork/orsirr_1.txt'

        DDG_G = np.loadtxt(DDG, dtype=int)
        DEL_G = np.loadtxt(DEL, dtype=int)
        DWT5_G = np.loadtxt(DWT5, dtype=int)
        DWT7_G = np.loadtxt(DWT7, dtype=int)
        LSH_G = np.loadtxt(LSH, dtype=int)
        ORS_G = np.loadtxt(ORS, dtype=float)
        ORS_G = ORS_G[:,0:2]

        DDG_G_nodes = max(max(DDG_G[:,0]), max(DDG_G[:,1]))
        DEL_G_nodes = max(max(DEL_G[:, 0]), max(DEL_G[:, 1]))
        DWT5_G_nodes = max(max(DWT5_G[:, 0]), max(DWT5_G[:, 1]))
        DWT7_G_nodes = max(max(DWT7_G[:, 0]), max(DWT7_G[:, 1]))
        LSH_G_nodes = max(max(LSH_G[:, 0]), max(LSH_G[:, 1]))
        ORS_G_nodes = int(max(max(ORS_G[:, 0]), max(ORS_G[:, 1])))

        row_indices1 = DDG_G[:, 0] - 1  
        col_indices1 = DDG_G[:, 1] - 1
        data1 = np.ones(len(DDG_G), dtype=int)  

        row_indices2 = DEL_G[:, 0] - 1
        col_indices2 = DEL_G[:, 1] - 1
        data2 = np.ones(len(DEL_G), dtype=int)

        row_indices3 = DWT5_G[:, 0] - 1
        col_indices3 = DWT5_G[:, 1] - 1
        data3 = np.ones(len(DWT5_G), dtype=int)

        row_indices4 = DWT7_G[:, 0] - 1
        col_indices4 = DWT7_G[:, 1] - 1
        data4 = np.ones(len(DWT7_G), dtype=int)

        row_indices5 = LSH_G[:, 0] - 1
        col_indices5 = LSH_G[:, 1] - 1
        data5 = np.ones(len(LSH_G), dtype=int)

        row_indices6 = ORS_G[:, 0] - 1
        col_indices6 = ORS_G[:, 1] - 1
        row_indices6 = row_indices6.astype(int)
        col_indices6 = col_indices6.astype(int)
        # data6 = ORS_G[:, 2]
        data6 = np.ones(len(ORS_G), dtype=int)

        DDG = scipy.sparse.csr_matrix((data1, (row_indices1, col_indices1)), shape=(DDG_G_nodes, DDG_G_nodes)).toarray()
        DEL = scipy.sparse.csr_matrix((data2, (row_indices2, col_indices2)), shape=(DEL_G_nodes, DEL_G_nodes)).toarray()
        DWT5 = scipy.sparse.csr_matrix((data3, (row_indices3, col_indices3)), shape=(DWT5_G_nodes, DWT5_G_nodes)).toarray()
        DWT7 = scipy.sparse.csr_matrix((data4, (row_indices4, col_indices4)), shape=(DWT7_G_nodes, DWT7_G_nodes)).toarray()
        LSH = scipy.sparse.csr_matrix((data5, (row_indices5, col_indices5)), shape=(LSH_G_nodes, LSH_G_nodes)).toarray()
        ORS = scipy.sparse.csr_matrix((data6, (row_indices6, col_indices6)), shape=(ORS_G_nodes, ORS_G_nodes)).toarray()


        with torch.no_grad():
            sub_result = np.zeros((1, 6))
            for sort, Graph in enumerate([DDG, DEL, DWT5, DWT7, LSH, ORS]):  #
                    result, ground_true_line = [], []
                    for i in range(10):
                        A = random_delete_nodes(Graph)
                        targer =  calculate_controllability_curve_corrected(nx.from_numpy_array(A), 'random')[:-1]
                        hypergraph_khop = generate_khop_hypergraph(A, 3)
                        hypergraph = torch.from_numpy(hypergraph_khop).cuda()
                        BC = compute_BC(A)
                        BC = torch.FloatTensor(BC).cuda()  
                        A = torch.FloatTensor(A).cuda()
                        BC = torch.unsqueeze(BC, dim=0)

                        output = self.forward(A, hypergraph, A, hypergraph,  BC = BC)
                        outputs = trans_to_cpu(torch.squeeze(output)).numpy()

                        print(f'Graph Type{sort}, Node ave_degrees{i}, Error value{i}={np.mean(np.abs(outputs-targer))}', end='\n')

                        result.append(outputs)
                        ground_true_line.append(targer)

                    result = np.array(result)  # 100 999
                    ground_true_line = np.array(ground_true_line)

                    average_std = np.mean(np.std(np.abs(result-ground_true_line), axis=0))
                    average_error = np.mean(np.mean(np.abs(result-ground_true_line), axis=1))
                    sub_result[0, sort] = average_error
                    if Exp_result_error[0, sort] > average_error:
                        Exp_result_error[0, sort] = average_error
                        Exp_result_sd[0, sort] = average_std

                        save_ground_true_line[f'{sort}'] = ground_true_line
                        save_result[f'{sort}'] = result

        save_result_fold = '/home/wu/The_second_project/PCR(GAT)/result'
        save_path1 = os.path.join(save_result_fold, f'TestM_realworld_ground_true_line.mat')
        save_path2 = os.path.join(save_result_fold, f'TestM_realworld_predicting_line.mat')
        scipy.io.savemat(save_path1, save_ground_true_line)
        scipy.io.savemat(save_path2, save_result)

        return Exp_result_error, Exp_result_sd, sub_result, save_ground_true_line, save_result