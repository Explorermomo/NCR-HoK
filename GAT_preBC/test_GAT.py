import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from time import time
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

# ======= 模型结构 =======
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
        out = self.fc(x)  # shape [N, 3]
        return out

# ======= 模型加载 =======
center_model = GATEncoder()
center_model.load_state_dict(
    torch.load(
        r'D:\BaiduNetdiskDownload\code_and_data\The_second_project\saved_models_by_type\ER\structure_encoder.pt',
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
center_model.to(device)
center_model.eval()

# ======= 计算函数 =======
def compute_BC(adj0):
    adj = adj0.cpu().numpy() if torch.is_tensor(adj0) else adj0
    G = nx.Graph(adj)
    start = time()
    betweeness_dict = nx.betweenness_centrality(G)
    BC = np.array([betweeness_dict[node] * 100 for node in range(len(betweeness_dict))])
    duration = time() - start
    return BC, duration

def predict_BC(adj):
    if isinstance(adj, np.ndarray):
        adj = csr_matrix(adj)
    edge_index, _ = from_scipy_sparse_matrix(adj)
    x = torch.ones((adj.shape[0], 1))
    data = Data(x=x, edge_index=edge_index).to(device)

    start = time()
    with torch.no_grad():
        pred = center_model(data)
    duration = time() - start
    return pred.cpu().numpy(), duration
  # <-- 加 * 100


# ======= 多规模测试 =======
node_sizes = [800, 1000, 1200]

print("===== 中心性时间与误差对比 =====")
for N in node_sizes:
    M = 5 * N
    G = nx.gnm_random_graph(N, M)
    adj = nx.adjacency_matrix(G)

    print(f"\n--- 图规模: {N} 节点, {M} 边 ---")

    # 精确中心性（networkx）
    bc_true, t1 = compute_BC(adj)

    # GAT 预测中心性
    bc_pred, t2 = predict_BC(adj)

    # === 误差计算 ===
    mae = np.mean(np.abs(bc_true - bc_pred))
    mape = np.mean(np.abs((bc_true - bc_pred) / (bc_true + 1e-8))) 


    print(f"传统中心性计算时间: {t1:.4f} 秒")
    print(f"GAT预测时间:         {t2:.4f} 秒")


    print(f"平均绝对误差 (MAE):   {mae:.4f}")
    print(f"平均相对误差 (MAPE):  {mape:.2f}%")

