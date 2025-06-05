import numpy as np

def levensthein_distance(a1,a2):
    dist=0
    a1 = a1.copy()
    a2 = a2.copy()
    sorted_idxA = np.argsort(a1[0])
    sorted_idxB = np.argsort(a2[0])
    a1[1] = a1[1][sorted_idxA]
    a2[1] = a2[1][sorted_idxB]
    a1[1] = a1[1].astype(int)
    a2[1] = a2[1].astype(int)
    for i in range(len(a1[1])):
        if a1[1][i] != a2[1][i]:
            dist+=(1-float(i)/len(a1[1]))
    return dist #/ ((len(a1[0]) + 1 ) / 2)


import numpy as np

def count_weighted_inversions(arr):
    n = len(arr)
    weights = [1 for i in range(1,len(arr))]
    #weights = [1-i/len(arr) for i in range(1,len(arr))]


    def merge_count_split_inv(left, right, left_indices, right_indices, start_left, start_right, weights):
        merged, merged_indices = [], []
        i, j, inv_count = 0, 0, 0.0
        
        already_moved = []
        while i < len(left) and j < len(right):

            if left[i] <= right[j]:
                merged.append(left[i])
                merged_indices.append(left_indices[i])
                i += 1
            else:
                merged.append(right[j])
                merged_indices.append(right_indices[j])
                
                
                # Calcola le posizioni globali reali
                pos_i = start_left + i + already_moved.count(left[i])
                pos_j = start_right + j

                already_moved.extend(left[i:])
                
                for k in range(pos_i, pos_j):
                    inv_count += weights[k]
                j += 1
                

        merged += left[i:]
        merged_indices += left_indices[i:]
        merged += right[j:]
        merged_indices += right_indices[j:]


        return merged, merged_indices, inv_count

    def sort_and_count(arr, indices, start_index):
        if len(arr) <= 1:
            return arr, indices, 0.0

        mid = len(arr) // 2
        left, left_indices, left_inv = sort_and_count(arr[:mid], indices[:mid], start_index)
        right, right_indices, right_inv = sort_and_count(arr[mid:], indices[mid:], start_index + mid)
        
        merged, merged_indices, split_inv = merge_count_split_inv(left, right, left_indices, right_indices, start_index, start_index + mid, weights)
        return merged, merged_indices, left_inv + right_inv + split_inv

    indices = list(range(n))
    merged, _, inv_count = sort_and_count(arr, indices, 0)  
    return inv_count

def perm_distance(a1, a2):
    a1 = a1.copy()
    a2 = a2.copy()
    sorted_idxA = np.argsort(a1[0])
    sorted_idxB = np.argsort(a2[0])
    seq1 = a1[1][sorted_idxA]
    seq2 = a2[1][sorted_idxB]


    seq1 = seq1.astype(int)
    seq2 = seq2.astype(int)
    index_map = {value: i for i, value in enumerate(seq2)}
    perm = [index_map[value] for value in seq1]
    num_transpositions = count_weighted_inversions(perm)
    
    return num_transpositions
