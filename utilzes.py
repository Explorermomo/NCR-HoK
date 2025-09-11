import torch
import numpy as np
import networkx as nx
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_distances as cos_dis, euclidean_distances

from model import *

# construct hypergraph by using K-Hop method
def generate_khop_hypergraph(adj0, k, have_glad=False):
    adj = adj0.cpu().numpy() if torch.is_tensor(adj0) else adj0
    graph = nx.Graph(adj)
    hyperedges = []
    N = adj.shape[0]
    for node in graph.nodes():
        neighbors = set(nx.single_source_shortest_path(graph, node, cutoff=k).keys())
        hyperedges.append(list(neighbors))
    hypergraph_adj = np.zeros((N,len(hyperedges)), dtype=int)
    for i, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            hypergraph_adj[node, i] = 1

    hypergraph_adj = torch.from_numpy(hypergraph_adj) if torch.is_tensor(adj0) else hypergraph_adj

    return hypergraph_adj

# construct  hypergraph by using K-NN
def generate_knn_hypergraph(adj0, k, X=None):
    adj = adj0.detach().cpu().numpy() if torch.is_tensor(adj0) else adj0
    hyperedges = []
    graph = nx.Graph(adj)
    N = adj.shape[0]
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= k:
            # hyperedge = tuple(neighbors[:k])
            hyperedge = [node] + neighbors[:k]
            hyperedges.append(hyperedge)

    hypergraph_adj = np.zeros((N,N), dtype=int)
    for i, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            hypergraph_adj[node, i] = 1

    hypergraph_adj = torch.from_numpy(hypergraph_adj) if torch.is_tensor(adj0) else hypergraph_adj

    return hypergraph_adj

# other
def construct_H_with_KNN(X0, K_neigs = 10, is_probH=False, m_prob=1):
    X = X0.cpu().numpy() if torch.is_tensor(X0) else X0
    # if len(X.shape !=2 ):
    #     X = X.reshape(-1, X.shape[-1])
    # X = X0
    # X = X.cpu()
    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dia_mat = cos_dis(X)
    H = None
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distence(dia_mat, k_neig, is_probH, m_prob)
        H = hyperedge_concat(H, H_tmp)

    hypergraph_adj = torch.from_numpy(H) if torch.is_tensor(X0) else H
    return hypergraph_adj

def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None:
            if H is None:
                H = h
            else:
                H = np.hstack((H,h))

    return H

def construct_H_with_KNN_from_distence(dis_mat, k_neig, is_probH=False, m_prob=1):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx,center_idx]=0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig]==center_idx):
            nearest_idx[k_neig-1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0,node_idx]**2/(m_prob*avg_dis)**2)
            else:
                H[node_idx,center_idx] = 1
    return H

def sampled_idx(ids, k):
    df = pd.DataFrame(ids)
    sample_ids = df.sample(k-1, replace = True).values
    sample_ids = sample_ids.flatten().tolist()
    sample_ids.append(ids[-1])
    return sample_ids

def construct_edge_list_from_Kmeans(X0, clusters, adjacent_clusters, k_neighbors) -> np.array:
    # X = X.cpu()
    X = X0.cpu().numpy() if torch.is_tensor(X0) else X0
    N = X.shape[0]
    kmeans = KMeans(n_clusters = clusters, random_state=42).fit(X)
    centers = kmeans.cluster_centers_
    dis = euclidean_distances(X, centers)
    _, clusters_center_dict = torch.topk(torch.Tensor(dis), adjacent_clusters, largest=False)
    point_labels = kmeans.labels_
    point_labels_in_which_cluster = [np.where(point_labels==i)[0] for i in range(clusters)]

    def lis_cat(list_of_array):
        ret = list()
        for array in list_of_array:
            ret += array.tolist()
        return ret

    cluster_neighbor_dict = [lis_cat([point_labels_in_which_cluster[clusters_center_dict[point][i]] for i in range(adjacent_clusters)]) for point in range(N)]
    for point,entry in enumerate(cluster_neighbor_dict):
        entry.append(point)
    sampled_ids = [sampled_idx(cluster_neighbor_dict[point], k_neighbors) for point in range( N)]
    sampled_ids = np.array(sampled_ids)

    hypergraph_adj = np.zeros((sampled_ids.shape[0], sampled_ids.shape[0]))
    for i in range(sampled_ids.shape[0]):
        hypergraph_adj[i][sampled_ids[i, :]] = 1
    hypergraph_adj = torch.from_numpy(hypergraph_adj) if torch.is_tensor(X0) else hypergraph_adj
    return hypergraph_adj.T



# me
def generate_knn_hypergraph_by_feature(adj,  k, X=None):

    adj = np.array(adj.cpu())

    N = adj.shape[0]
    hypergraph_adj = np.zeros((N,N), dtype=int)

    for i in range(N):
        node_distance = []
        for j in range(N):
            if i != j and adj[i][j] == 1:
                distance = np.linalg.norm(X[i]-X[j])
                node_distance.append((j, distance))
        node_distance.sort(key=lambda x: x[1])
        k_nearest_neighbors = [node for node, _ in node_distance[:k]]
        for neighbor in k_nearest_neighbors:
            hypergraph_adj[neighbor][i] = 1

    hypergraph_adj = torch.tensor(hypergraph_adj, dtype=int).cuda()

    return hypergraph_adj



# construct hypergraph by using k-shell method
def generate_kshell_hypergraph(adj0, X=None):
    # adj = np.array(adj.cpu())
    adj = adj0.cpu().numpy() if torch.is_tensor(adj0) else adj0
    N = adj.shape[0]
    G = nx.Graph(adj)

    k_shells = nx.onion_layers(G)
    # k_shells = nx.onion_layers(G)
    max_k_shell = max(k_shells.values())

    hypergraph_adj = np.zeros((N, max_k_shell), dtype=int)
    for k in range(1, max_k_shell+1):
        shell_nodes = [node for node in k_shells if k_shells[node] == k]

        for node in shell_nodes:
            hypergraph_adj[node, k-1] = 1

    hypergraph_adj = torch.from_numpy(hypergraph_adj) if torch.is_tensor(adj0) else hypergraph_adj

    return hypergraph_adj

# generate hypergraph by using kmeans method
def generate_kmeans_hypergraph(X, n_clusters):

    X = np.array(X.cpu())

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    cluster_assignments = kmeans.labels_
    N = X.shape[0]

    hypergraph_adj = np.zeros((N, N), dtype=int)

    for node in range(N):
        node_cluster = cluster_assignments[node]
        cluster_nodes = np.where(cluster_assignments == node_cluster)[0]

        for neighbor in cluster_nodes:
            hypergraph_adj[node, neighbor] = 1

    # hypergraph_adj = torch.tensor(hypergraph_adj, dtype=int).cuda()

    return hypergraph_adj


def compute_BC(adj0):
    adj = adj0.cpu().numpy() if torch.is_tensor(adj0) else adj0
    BC = []
    G = nx.Graph(adj)
    betweeness_dict = nx.betweenness_centrality(G)
    for node in betweeness_dict:
        BC.append(betweeness_dict[node]*100)

    # BC = torch.FloatTensor(BC).cuda()
    BC = np.array(BC)
    return BC















