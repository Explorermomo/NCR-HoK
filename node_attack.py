import networkx as nx
import numpy as np
import random
from copy import deepcopy
from typing import Union


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

        # N_current = len(G.nodes())
        # # print(N_current)
        # ND = max(1, N_current - rank_A)
        # # print(ND)
        # nD = ND / (N_current-i)

        controllability_curve.append(r_i)

    return controllability_curve


def node_attack(graph: Union[nx.Graph, nx.DiGraph], strategy: str = 'degree') -> list:
    """
        node attacks, under a certain strategy.

        Parameters
        ----------
        graph : the graph to be attacked
        strategy : the strategy of choosing targets under node removals

        Returns
        -------
        the attack sequence of nodes
    """
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
