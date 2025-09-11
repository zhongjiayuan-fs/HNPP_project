import gudhi
import numpy as np
import numpy as np
from scipy.stats import pearsonr
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


df = pd.read_csv("Pericyte_to_neuron.csv",index_col=0,header=None) 
data_value = df.values
data_value=np.log(data_value+1)

TF_df = pd.read_csv("TF_human.csv",index_col=0,header=None)
TF_df_list =TF_df.index

data_gene_list=df.index
TF_result = pd.Series(0, index=data_gene_list)
TF_result.loc[df.index.isin(TF_df.index)] = 1
TF_result_list = TF_result.tolist()
TF_result_array = np.array(TF_result_list)
sum_values = np.sum(TF_result_array)
if sum_values == 0:
    normalized_array = np.zeros_like(TF_result_array)/len(TF_result_array)  # 赋予均匀分布
else:
    normalized_array = TF_result_array / sum_values

cell_num =[76,86,48,283,61,69]

cell_count=0;
PR_mean_value = []
for k in range(len(cell_num)):
    after_cell_count = sum(cell_num[0:k+1])
    data_matrix = data_value[:, cell_count:after_cell_count]
    cell_count = sum(cell_num[0:k+1])
    data_matrix_size = data_matrix.shape[0]
    p_values = np.ones((data_matrix_size, data_matrix_size))
    corr_matrix = np.zeros((data_matrix_size, data_matrix_size))
    for i in range(data_matrix_size):
        for j in range(data_matrix_size):
            corr, p_value = pearsonr(data_matrix[i, :], data_matrix[j, :])
            p_values[i, j] = p_value
            corr_matrix[i, j] = corr
    corr_matrix = np.nan_to_num(corr_matrix)
    p_values = np.nan_to_num(p_values, nan=1)
    corr_matrix[p_values > 0.05] = 0
    correlation_matrix = abs(corr_matrix)
    distance_matrix = 1 - correlation_matrix
    
    max_edge_length_threshold =0.2 
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length_threshold)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    edge_2_simplices_counts = {}
    for simplex in simplex_tree.get_skeleton(1):
        if len(simplex[0]) == 2:         
            edge = tuple(sorted(simplex[0]))
            edge_2_simplices_counts[edge] = 0
            
    for simplex in simplex_tree.get_skeleton(2):
        if len(simplex[0]) == 3:   
            vertices = simplex[0]
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([vertices[i], vertices[j]]))
                    if edge in edge_2_simplices_counts:
                        edge_2_simplices_counts[edge] += 1
                        
    num_nodes = correlation_matrix.shape[0]
    node_2_simplices_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge, count in edge_2_simplices_counts.items():
        node1, node2 = edge
        node_2_simplices_matrix[node1, node2] = count
        node_2_simplices_matrix[node2, node1] = count      
    
    total_outdegree = np.sum(node_2_simplices_matrix, axis=1)
    transition_matrix = node_2_simplices_matrix / (total_outdegree[:, None] + 0.0000001)
    
    
    damping_factor = 0.85
    factor_1=0.1
    factor_2=1-damping_factor-factor_1
    N = node_2_simplices_matrix.shape[0]
    initial_PR = np.ones(N) / N
    PR = initial_PR
    epsilon = 1e-6
    zero_rows_mask = np.all(transition_matrix == 0, axis=1)
    row_vector = np.where(zero_rows_mask, 1, 0)
    one_vector = np.ones(len(row_vector))
    Outer_product = np.outer(row_vector, one_vector)
    Outer_product_matrix =Outer_product / N
    personalized_vector = np.std(data_matrix, axis=1)
    sum_values = np.sum(personalized_vector)
    if sum_values == 0:
       normalized_vector = np.zeros_like(personalized_vector) / len(personalized_vector)  
    else:
       normalized_vector = personalized_vector / sum_values
    result_matrix = transition_matrix + Outer_product_matrix
   
    while True:
        new_PR = damping_factor * np.dot(result_matrix.T, PR) + factor_2 * normalized_array + factor_1 * normalized_vector
        if np.linalg.norm(new_PR - PR, 1) < epsilon: 
            break
        PR = new_PR
    PR=PR/np.mean(PR)  
    sorted_indices = np.argsort(PR)[::-1]
    index = int(len(PR) * (5 / 100))
    PR_mean_value.append(np.mean(PR[sorted_indices[:index]]))
    print(k)
    print(PR_mean_value)


labels = ['0', '1', '2', '3','4','5'] 
plt.plot(labels, PR_mean_value, marker='o', linestyle='-')
plt.xlabel('Parameter')
plt.ylabel('High-order PageRank')
plt.grid(True)
plt.show()