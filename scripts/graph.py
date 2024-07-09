from typing import List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components



def pairs_to_connected_components(pairs: List[Tuple[int, int]], graph_size: int) -> List[List[int]]:
    # Build the sparse adjacency matrix
    data = np.ones(len(edges))
    row_indices, col_indices = zip(*pairs)
    adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(graph_size, graph_size))

    n_components, component_labels = connected_components(adjacency_matrix)

    result = []
    for i in range(n_components):
        result.append(np.argwhere(component_labels == i).flatten())

    return result

# Example list of edges
edges = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (9, 7)]

result = pairs_to_connected_components(edges, 10)
print(result)
