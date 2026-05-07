import os
import gudhi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

df = pd.read_csv("Pericyte_to_neuron.csv",index_col=0,header=None) 

data_value = df.values.astype(float)
data_value = np.log1p(data_value)

TF_df = pd.read_csv("TF_human_unique.csv", index_col=0, header=None)


data_gene_list = df.index
TF_result = pd.Series(0, index=data_gene_list)
TF_result.loc[df.index.isin(TF_df.index)] = 1
TF_result_array = TF_result.to_numpy(dtype=float)

sum_values = np.sum(TF_result_array)
if sum_values == 0:
    normalized_array = np.zeros_like(TF_result_array) / len(TF_result_array)
else:
    normalized_array = TF_result_array / sum_values

print(normalized_array)
print("Normalized Array:", normalized_array)

cell_num = [76, 86, 48, 283, 61, 69]  


max_edge_length_threshold =0.2 

damping_factor = 0.85
factor_1 = 0.1
factor_2 = 1 - damping_factor - factor_1
epsilon = 1e-6


def fast_corr_pvalue_matrix(data_matrix):

    n_genes, n_cells = data_matrix.shape


    corr_matrix = np.corrcoef(data_matrix)

    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = corr_matrix * np.sqrt((n_cells - 2) / (1.0 - corr_matrix ** 2))
        p_values = 2 * t.sf(np.abs(t_stat), df=n_cells - 2)

 
    corr_matrix = np.nan_to_num(corr_matrix)
    p_values = np.nan_to_num(p_values, nan=1.0)

    return corr_matrix, p_values


PR_mean_value = []

cum_cell_num = np.cumsum(cell_num)
start_idx = 0

for k, end_idx in enumerate(cum_cell_num):

    data_matrix = data_value[:, start_idx:end_idx]
    start_idx = end_idx

    num_nodes = data_matrix.shape[0]


    corr_matrix, p_values = fast_corr_pvalue_matrix(data_matrix)

    corr_matrix[p_values > 0.05] = 0

    correlation_matrix = np.abs(corr_matrix)
    distance_matrix = 1 - correlation_matrix
    
    rips_complex = gudhi.RipsComplex(
        distance_matrix=distance_matrix,
        max_edge_length=max_edge_length_threshold
    )
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)


    node_2_simplices_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    for simplex, _ in simplex_tree.get_skeleton(2):
        if len(simplex) == 3:
            a, b, c = simplex
            node_2_simplices_matrix[a, b] += 1
            node_2_simplices_matrix[b, a] += 1

            node_2_simplices_matrix[a, c] += 1
            node_2_simplices_matrix[c, a] += 1

            node_2_simplices_matrix[b, c] += 1
            node_2_simplices_matrix[c, b] += 1


    total_outdegree = np.sum(node_2_simplices_matrix, axis=1)
    transition_matrix = node_2_simplices_matrix / (total_outdegree[:, None] + 1e-7)


    N = num_nodes
    PR = np.ones(N) / N


    zero_rows_mask = np.all(transition_matrix == 0, axis=1)
    row_vector = np.where(zero_rows_mask, 1, 0)
    one_vector = np.ones(len(row_vector))
    Outer_product_matrix = np.outer(row_vector, one_vector) / N


    personalized_vector = np.std(data_matrix, axis=1)
    sum_values = np.sum(personalized_vector)
    if sum_values == 0:
        normalized_vector = np.zeros_like(personalized_vector) / len(personalized_vector)
    else:
        normalized_vector = personalized_vector / sum_values

    result_matrix = transition_matrix + Outer_product_matrix


    while True:
        new_PR = (
            damping_factor * (result_matrix.T @ PR)
            + factor_2 * normalized_array
            + factor_1 * normalized_vector
        )

        if np.linalg.norm(new_PR - PR, 1) < epsilon:
            PR = new_PR
            break

        PR = new_PR

    PR = PR / np.mean(PR)

    sorted_indices = np.argsort(PR)[::-1]

    index = int(len(PR) * (5 / 100))

    PR_mean_value.append(np.mean(PR[sorted_indices[:index]]))

    print(k)
    print(PR_mean_value)

labels = ['0', '1', '2', '3', '4', '5']   

plt.plot(labels, PR_mean_value, marker='o', linestyle='-')
plt.xlabel('Parameter')
plt.ylabel('HNPP')
plt.grid(True)
plt.savefig("HNPP.png", dpi=300, bbox_inches='tight') 
plt.show()
