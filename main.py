import time
import os
import random
import warnings
import torch
import numpy as np
import argparse
import scipy.io as sio
import scipy
import scipy.sparse as sps
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from data_process import *
from model import *

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--initialFeatureSize', type=int, default=30, help='initial size')
parser.add_argument('--outputFeatureSize', type=int, default=10, help='initial size')
parser.add_argument('--EndNodeFeature', type=int, default=2, help='initial size')
parser.add_argument('--n_hid', type=int, default=60, help='hidden state size')
parser.add_argument('--degree_size', type=int, default=10, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--fc_dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=2, help='the number of steps after which the learning rate decay')  # 改为1
parser.add_argument('--l2', type=float, default=1e-6, help='l2 penalty')
parser.add_argument('--rand', type=int, default=42, help='rand_seed')
parser.add_argument('--clip_value', type=float, default=0.8, help='the gradient clip')


# the initial setting of this work
parser.add_argument('--batch_size', type=int, default=5)

device = torch.device('cuda')
args = parser.parse_args()
print(args)

SEED = args.rand
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
def save_model(model, optimizer, epoch, avg_loss, best_error, exp_error, exp_sd, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'best_error': best_error,
        'exp_result_error': exp_error,
        'exp_result_sd': exp_sd,
        'args': args
    }, filepath)

def load_model(model, optimizer, filepath):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'✅ Loaded model from {filepath}')
        print(f'   Epoch: {checkpoint["epoch"]}, Best Error: {checkpoint["best_error"]:.4f}')
        return checkpoint
    else:
        print(f'❌ Model file {filepath} not found!')
        return None


### initial settinf
def main():
    global_loss = float('inf')
    cur_path = os.path.dirname(os.path.abspath(__file__))  
    
    aType = 'td'  # attack method 'ra' = rand attack;  'tb' = tar betweenness attack;  'td' = tar degree attack
    # train_val_mat = sio.loadmat(aType + '.mat')  # load training data
    train_val_mat = sio.loadmat(os.path.join(cur_path, aType + '.mat'))  # load training data
    data_train = train_val_mat['data_train']
    all_graph_number = data_train.shape[0]

    # shuffle_list_train = random.sample(range(0, all_graph_number), 20)
    # for i in shuffle_list_train:
    #     print(data_train[i, 0]['b'][0][0][0])

    grouded_indiecs = {}
    for i in range(all_graph_number):
        each_graph = data_train[i, 0]['b'][0][0][0]
        part1s = each_graph.split('-')[0]
        part2s = each_graph.split('ak')[1][0]
        keys = f'{part1s}_{part2s}'
        if keys not in grouded_indiecs:
            grouded_indiecs[keys] = [i]
        else:
            grouded_indiecs[keys].append(i)
    selected_indices = []
    for indices in grouded_indiecs.values():
        num_to_select = int(len(indices) * 0.02)
        selected_indices.extend(random.sample(indices, num_to_select))


    # construct the model
    model_dir = os.path.join(cur_path, 'saved_models_td')
    os.makedirs(model_dir, exist_ok=True)
    
   
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"💾 Model will be saved to: {model_dir}")
    # construct the model
    # model = trans_to_cuda(Robustness_predict_modul(args))
    Graph_node_number = 1000
    model = trans_to_cuda(Robustness_predict_modul(args, Graph_node_number, Graph_node_number-1))
    # checkpoint_path = os.path.join(model_dir, 'best_model_epoch_1_error_0.0296.pth')
    # checkpoint = load_model(model, model.optimizer, checkpoint_path) 
    checkpoint = None
    start_epoch = 0 
    if checkpoint:
        start_epoch = checkpoint['epoch']  
    

    # training model and output the result
    # e_average_min, sd_average_min= 1000000, 1000000
    Exp_result_error = np.ones((4, 4)) * 100000
    Exp_result_sd = np.ones((4, 4)) * 100000
    result = []
    
    best_overall_error = float('inf')
    best_epoch = 0
    best_model_path = None
    # epoch_pbar = tqdm(range(args.epoch), desc="Total Training")
    epoch_pbar = tqdm(range(start_epoch, args.epoch), desc="Total Training")

    save_result, save_ground_true_line = {}, {}

    epoch_time0 = time.time()
    Each_Epoch_Time = []
    Loss_Value,Each_Time = [],[]

    # fig, ax = plt.subplots(figsize=(10, 5))
    # save_result_fold = '/home/wu/The_second_project/PCR(GAT)/result/last_one_exp'

    for epoch in epoch_pbar:
        # train and val one batch by one batch
        epoch_loss_value, epoch_time,avg_loss = model.TrainM(train_val_mat,epoch=epoch, begin_time= epoch_time0, all_train_index= selected_indices)


        # # select the value of k-degree: 0 1 2 3 -> 2 5 8 10
        # k = 0
        # # select the value of topologies: 0 1 2 3 -> ER SF QSN SW
        # type = 0

        epoch_timeI = time.time()
        Exp_result_error, Exp_result_sd, sub_result, save_ground_true_line, save_result = model.TestM(train_val_mat, save_result, save_ground_true_line, EndEpoch=epoch, Exp_result_error= Exp_result_error, Exp_result_sd=Exp_result_sd)
        epoch_timeII = time.time()
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print((epoch_timeII-epoch_timeI)/1600)
        result.append(sub_result)
        best_error = np.min(Exp_result_error)
        current_avg_error = np.mean(sub_result)
        current_min_error = np.min(sub_result)
        print(f'\n📊 Epoch {epoch+1} Results:')
        print(f'   Training Loss: {avg_loss:.4f}')
        print(f'   Test Average Error: {current_avg_error:.4f}')
        print(f'   Test Min Error: {current_min_error:.4f}')
        print(f'   Best Overall Error: {best_overall_error:.4f}')
        # epoch_pbar.close()
        checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_model(model, model.optimizer, epoch+1, avg_loss, current_min_error, Exp_result_error, Exp_result_sd, checkpoint_path)
        print(f'💾 Saved checkpoint: {checkpoint_path}')

        if current_avg_error < best_overall_error:
            best_overall_error = current_avg_error
            best_epoch = epoch + 1
            
            best_model_path = os.path.join(model_dir, f'best_model_epoch_{best_epoch}_error_{best_overall_error:.4f}.pth')
            save_model(model, model.optimizer, best_epoch, avg_loss, best_overall_error, Exp_result_error, Exp_result_sd, best_model_path)
            print(f'🎯 NEW BEST MODEL! Saved: {best_model_path}')
            
            fixed_best_path = os.path.join(model_dir, 'best_model.pth')
            save_model(model, model.optimizer, best_epoch, avg_loss, best_overall_error, Exp_result_error, Exp_result_sd, fixed_best_path)

        print(f'{"="*60}')
    final_model_path = os.path.join(model_dir, 'final_model.pth')
    save_model(model, model.optimizer, args.epoch, avg_loss, current_min_error, Exp_result_error, Exp_result_sd, final_model_path)
    print(f'🏁 Saved final model: {final_model_path}')

    result_path = os.path.join(model_dir, 'training_results.npz')
    np.savez(result_path, 
            exp_result_error=Exp_result_error,
            exp_result_sd=Exp_result_sd,
            all_results=np.array(result),
            best_error=best_overall_error,
            best_epoch=best_epoch)
    
    print(f'📊 Saved results: {result_path}')    

    return Exp_result_error, Exp_result_sd, result


### main()
if __name__ == '__main__':
    e_average, sd_average = 0 , 0
    for i in range(1):
        e_average, sd_average, result = main()
    print("=============================== the result of error value ===========================", end='\n')
    print(e_average)
    print("=============================== the result of st value ===========================", end='\n')
    print(sd_average)
    print("=============================== the all result ===========================", end='\n')
    print(result)





