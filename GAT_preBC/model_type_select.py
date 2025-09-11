import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# MLP编码器（图结构不变，仅处理节点特征）
class MLPEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, out_dim=3):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = data.x  # 节点特征 [N, 1]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out  # 输出形状 [N, 3]

# GCN编码器
class GCNEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, out_dim=1):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        out = self.fc(x)
        return out  # 输出形状 [N, 3]

# GAT编码器
class GATEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, out_dim=1, heads=4):
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        out = self.fc(x)
        return out  # 输出形状 [N, 3]
