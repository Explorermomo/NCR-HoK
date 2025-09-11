# attack simulation example
# for run time comparison

import timeit
import numpy as np
import random
from networkx import nx

start = timeit.default_timer()
N = 1000    # number of nodes
M = 5*N     # number of edges
G = nx.gnm_random_graph(N, M, directed=True)  # generate the random graph
cv = []     # controllability curve
A = nx.adjacency_matrix(G).todense()
#print(np.linalg.matrix_rank(A))
cv.append(max(N-np.linalg.matrix_rank(A), 1)/N)


for i in range(N-1):
    nid = random.choice(list(G))
    G.remove_node(nid)
    A = nx.adjacency_matrix(G).todense()
    #print(np.linalg.matrix_rank(A))
    cv.append(max(G.number_of_nodes() - np.linalg.matrix_rank(A),1)/G.number_of_nodes())

#print(cv)


stop = timeit.default_timer()
print('Time: ', stop - start)

# average run time = 162.68s
# Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz 2.59GHz
# Installed memory (RAM): 16.0GB (15.8 usable)
# Windows 10 Home 64-bit Operating System

