import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
from dgl.nn import GINConv
from torch.utils.data import Dataset
import random
import numpy as np
import networkx as nx
import dgl
import random
from torch.utils.data import Dataset
import dgl
import torch
import scipy.io as sio

# 保存数据集到 .mat 文件
def save_dataset_to_mat(graphs, features, labels, filename):
    """
    Save the dataset (graphs, features, labels) into a .mat file.
    
    Parameters:
    - graphs: List of DGL graphs
    - features: List of feature tensors
    - labels: List of label tensors
    - filename: The path to save the .mat file
    """
    dataset = {
        'graphs': graphs,
        'features': features,
        'labels': labels
    }
    sio.savemat(filename, dataset)

# 从 .mat 文件加载数据集
def load_dataset_from_mat(filename):
    """
    Load the dataset (graphs, features, labels) from a .mat file.
    
    Parameters:
    - filename: The path to the .mat file containing the dataset.
    
    Returns:
    - graphs: List of DGL graphs
    - features: List of feature tensors
    - labels: List of label tensors
    """
    data = sio.loadmat(filename)
    
    # Assuming data['graphs'], data['features'], and data['labels'] are in the .mat file
    graphs = data['graphs']
    features = data['features']
    labels = data['labels']
    
    # Convert graphs to DGL graphs if necessary
    graphs = [dgl.from_networkx(nx.from_numpy_matrix(graph)) for graph in graphs]
    
    return graphs, features, labels

def collate(samples):
    graphs, features, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)  # 将多个图合并为一个大图
    batched_features = torch.cat(features, dim=0)  # 合并所有节点特征
    batched_labels = torch.stack(labels)  # 合并所有标签
    return batched_graph, batched_features, batched_labels

class GraphDataset(Dataset):
    def __init__(self, graphs, features, labels):
        """
        Args:
            graphs (list of DGL graphs): The list of DGL graphs
            features (list of torch tensors): The list of features for each graph
            labels (list of torch tensors): The list of labels (e.g., NCR_HOK)
        """
        self.graphs = graphs
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.features[idx], self.labels[idx]

def generate_synthetic_graph(graph_type='ER', num_nodes=100, attack_type='degree', **kwargs):
    """
    Generates a synthetic graph of the specified type.

    Parameters:
    - graph_type: Type of the graph to generate ('ER', 'SF', 'QSN', 'SW')
    - num_nodes: Number of nodes in the graph
    - attack_type: Attack strategy for label calculation (not used in graph generation)
    - kwargs: Additional parameters for specific graph types

    Returns:
    - dgl_graph: DGL graph representation
    - degrees: Node degrees in the graph
    - labels: Control robustness labels (calculated via `simulate_attack` or `calculate_controllability_labels`)
    """
    if graph_type == 'ER':  # Erdos-Renyi graph
        p = kwargs.get('p', 0.05)  # probability of edge creation
        g = nx.erdos_renyi_graph(num_nodes, p)
    elif graph_type == 'SF':  # Scale-Free graph
        g = nx.scale_free_graph(num_nodes)
        g = nx.Graph(g)  # Convert to undirected graph
        g.remove_edges_from(nx.selfloop_edges(g))
    elif graph_type == 'QSN':  # QSN (Small-World Network)
        k = kwargs.get('k', 4)  # Each node is connected to k nearest neighbors in a ring topology
        p = kwargs.get('p', 0.1)  # Probability of rewiring each edge
        g = nx.random_regular_graph(k, num_nodes)  # Generate a small world graph
        nx.set_edge_attributes(g, {e: random.random() for e in g.edges()}, "weight")
    elif graph_type == 'SW':  # Watts-Strogatz small-world model
        k = kwargs.get('k', 4)  # Each node is connected to k nearest neighbors in a ring topology
        p = kwargs.get('p', 0.1)  # Probability of rewiring each edge
        g = nx.watts_strogatz_graph(num_nodes, k, p)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    # Convert to DGL graph
    dgl_g = dgl.from_networkx(g)

    # Node degree feature
    degrees = dgl_g.in_degrees().view(-1, 1).float()

    # Use calculate_controllability_labels or simulate_attack to generate labels (for simplicity, use calculate_controllability_labels here)
    labels = calculate_controllability_labels(dgl_g, strategy=attack_type)  # Calculate labels based on controllability

    return dgl_g, degrees, labels

# MLP for GIN
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# GIN Layer
class GINLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GINLayer, self).__init__()
        self.mlp = MLP(input_dim, output_dim, output_dim)
        self.conv = GINConv(self.mlp, aggregator_type='sum')

    def forward(self, g, h):
        return self.conv(g, h)

# GIN-MAS Model with Single Output (NCR_HOK)
class GIN_MAS(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GIN_MAS, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(GINLayer(in_dim, hidden_dim))

        self.readout_mlp = nn.Linear(hidden_dim * num_layers, 1)  # Only one output for NCR_HOK

    def forward(self, g, h):
        all_outputs = []
        for layer in self.layers:
            h = layer(g, h)
            all_outputs.append(dgl.readout_nodes(g, h, op='sum'))

        hg = torch.cat(all_outputs, dim=1)
        return self.readout_mlp(hg)  # Only returns one value: NCR_HOK

# Multitask Loss with Uncertainty Weighting
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks=1):
        super(UncertaintyWeightedLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, preds, targets):
        loss = 0
        for i in range(preds.size(1)):
            precision = torch.exp(-self.log_vars[i])
            loss += precision * F.mse_loss(preds[:, i], targets[:, i]) + self.log_vars[i]
        return loss
from copy import deepcopy
from typing import Union

# Node attack function for controlling strategy
def node_attack(graph: Union[nx.Graph, nx.DiGraph], strategy: str = 'degree') -> list:
    sequence = []
    G = deepcopy(graph)
    N = G.number_of_nodes()
    for _ in range(N - 1):
        if strategy == 'degree':
            degrees = dict(nx.degree(G))
            _node = max(degrees, key=degrees.get)
        elif strategy == 'random':
            _node = random.sample(list(G.nodes), 1)[0]
        elif strategy == 'betweenness':
            bets = dict(nx.betweenness_centrality(G))
            _node = max(bets, key=bets.get)
        else:
            raise AttributeError(f'Attack strategy: {strategy}, NOT Implemented.')
        G.remove_node(_node)
        sequence.append(_node)
    return sequence

# Modified function to calculate controllability curve based on node attacks
def calculate_controllability_curve_corrected(G, strategy):
    N = G.number_of_nodes()
    attack_node_sequence = node_attack(graph=G, strategy=strategy)

    rank_A = np.linalg.matrix_rank(nx.to_numpy_array(G))
    r_0 = max(1, N - rank_A) / N
    controllability_curve = [r_0]

    for i, node in enumerate(attack_node_sequence):

        G.remove_node(node)
        A = nx.to_numpy_array(G)
        rank_A = np.linalg.matrix_rank(A)

        r_i = max(1, (N - i - 1) - rank_A) / (N - i - 1)

        controllability_curve.append(r_i)

    return controllability_curve

# Replace simulate_attack to use the controllability curve as labels
def calculate_controllability_labels(graph, strategy='degree'):
    controllability_curve = calculate_controllability_curve_corrected(graph, strategy)
    # For simplicity, return the final controllability score
    return torch.tensor([controllability_curve[-1]], dtype=torch.float)  # Only return the last value

# GraphDataset class to load and manage data
class GraphDataset(Dataset):
    def __init__(self, graphs, features, labels):
        self.graphs = graphs
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.features[idx], self.labels[idx]
from dgl.data import GINDataset
# Modify loading function to calculate controllability labels instead of previous labels
def load_real_dataset(num_graphs, attack_type='degree'):

    dataset = GINDataset(name='REDDIT-MULTI-12K', self_loop=True)
    graphs = []
    features = []
    labels = []
    for g in dataset.graphs[:num_graphs]:
        dgl_g = g
        degrees = dgl_g.in_degrees().view(-1, 1).float()
        label = calculate_controllability_labels(dgl_g, strategy=attack_type)  # Use controllability curve
        graphs.append(dgl_g)
        features.append(degrees)
        labels.append(label)
    return graphs, features, labels

# Training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batched_graph, features, labels in dataloader:
        batched_graph = batched_graph.to(device)
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(batched_graph, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batched_graph, features, labels in dataloader:
            batched_graph = batched_graph.to(device)
            features, labels = features.to(device), labels.to(device)
            outputs = model(batched_graph, features)
            total_loss += F.mse_loss(outputs, labels, reduction='sum').item()
    return total_loss / len(dataloader.dataset)
from dgl.dataloading import GraphDataLoader
from torch.utils.data import random_split

# Main training function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成数据集并保存
    print("Generating synthetic dataset with random attack...")
    graphs, features, labels = [], [], []
    for _ in range(1000):
        g, f, l = generate_synthetic_graph(graph_type='BA', num_nodes=random.randint(500, 1000), m=5, attack_type='random')
        graphs.append(g)
        features.append(f)
        labels.append(l)

    # 保存数据集到本地
    save_dataset_to_mat(graphs, features, labels, 'dataset.mat')

    # 加载保存的数据集
    graphs, features, labels = load_dataset_from_mat('dataset.mat')

    dataset = GraphDataset(graphs, features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = GraphDataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = GraphDataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)

    model = GIN_MAS(input_dim=1, hidden_dim=64, num_layers=5).to(device)
    criterion = UncertaintyWeightedLoss(num_tasks=1)  # Only one task for NCR_HOK
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    for epoch in range(1, 21):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    main()
