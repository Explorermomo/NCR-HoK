import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from tqdm import tqdm
import random
import os

def generate_graph(graph_type, num_nodes, k_degree=5):
    num_edges = num_nodes * k_degree // 2  # 无向图，边数是节点数×平均度除2

    if graph_type == 'ER':
        # gnm_random_graph按节点数和边数生成，更稳定控制度
        G = nx.gnm_random_graph(num_nodes, num_edges)
    elif graph_type == 'SF':
        # Scale-Free图用Barabasi-Albert，指定每个新节点连多少条边
        # BA模型的度是2 * m，m是新节点连边数量
        m = max(1, k_degree // 2)  # 保证m至少是1
        G = nx.barabasi_albert_graph(num_nodes, m)
    elif graph_type == 'SW':
        # Small World用Watts-Strogatz
        k = max(2, (k_degree // 2) * 2)  # k必须是偶数，最少2
        G = nx.watts_strogatz_graph(num_nodes, k=k, p=0.1)
    elif graph_type == 'QSN':
        # 模拟QSN，用Watts-Strogatz低重连概率
        k = max(2, (k_degree // 2) * 2)
        G = nx.watts_strogatz_graph(num_nodes, k=k, p=0.01)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    degree = np.array([G.degree(n) for n in G.nodes()])
    x = torch.tensor(degree, dtype=torch.float).unsqueeze(1)

    bc_dict = nx.betweenness_centrality(G, normalized=True)
    bc = np.array([bc_dict[n] for n in G.nodes()])
    y = torch.tensor(bc, dtype=torch.float).unsqueeze(1)

    data = from_networkx(G)
    data.x = x
    data.y = y
    data.graph_type = graph_type
    return data


def generate_dataset_per_type(graph_type, num_graphs, min_nodes=800, max_nodes=1600, k_degrees=[2, 4, 5, 8]):
    dataset = []
    for _ in tqdm(range(num_graphs), desc=f"{graph_type} graphs"):
        num_nodes = random.randint(min_nodes, max_nodes)
        k_degree = random.choice(k_degrees)  
        data = generate_graph(graph_type, num_nodes, k_degree=k_degree)
        dataset.append(data)
    return dataset


def generate_all_datasets(save_dir, num_per_type=4000):
    os.makedirs(save_dir, exist_ok=True)
    all_data = []
    for graph_type in ['ER', 'SF', 'QSN', 'SW']:
        data = generate_dataset_per_type(graph_type, num_graphs=num_per_type)
        torch.save(data, os.path.join(save_dir, f'{graph_type}_graphs.pt'))
        all_data += data
    # 合并数据集
    torch.save(all_data[:int(len(all_data)*0.8)], os.path.join(save_dir, 'train_graphs.pt'))
    torch.save(all_data[int(len(all_data)*0.8):int(len(all_data)*0.9)], os.path.join(save_dir, 'val_graphs.pt'))
    torch.save(all_data[int(len(all_data)*0.9):], os.path.join(save_dir, 'test_graphs.pt'))

if __name__ == '__main__':
    save_path = r'D:\BaiduNetdiskDownload\code_and_data\The_second_project'
    generate_all_datasets(save_path, num_per_type=4000)
    print("✅ 图数据集已成功生成并保存在：", save_path)
