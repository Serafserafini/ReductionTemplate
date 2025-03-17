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
            dist+=1-float(i)/len(a1[1])
    return dist / ((len(a1[0]) + 1 ) / 2)

def dist1(a1,a2):
    dist=0
    for i in range(len(a1[0])):
        dist += ((a1[0].min() - a1[0][i])/(a1[0].min() - a1[0].max()) - (a2[0].min() - a2[0][i])/(a2[0].min() - a2[0].max())) ** 2
    return (dist/len(a1[0])) ** 0.5
# DISTANZA ENTALPIE E ORDINE 
def dist2(a1,a2):
    dist=0
    a1 = a1.copy()
    a2 = a2.copy()
    sorted_idxA = np.argsort(a1[0])
    sorted_idxB = np.argsort(a2[0])
    a1[1] = a1[1][sorted_idxA]
    a2[1] = a2[1][sorted_idxB]
    for i in range(len(a1[1])):
        if a1[1][i] != a2[1][i]:
            dist+=(1-float(i)/len(a1[1])) * (    abs(a1[0, int(a1[1,i]) ] - a1[0, int(a2[1,i]) ])/ (a1[0].max() - a1[0].min())    +   abs(a2[0, int(a1[1,i]) ] - a2[0, int(a2[1,i]) ])/ (a2[0].max() - a2[0].min())   ) / 2
    return dist / ((len(a1[0]) + 1 ) / 2)
# DISTANZA ORDINE E PESO SULLO SHIFT
def dist3 (a1, a2):
    dist = 0
    a1 = a1.copy()
    a2 = a2.copy()
    sorted_idxA = np.argsort(a1[0])
    sorted_idxB = np.argsort(a2[0])
    a1[1] = a1[1][sorted_idxA]
    a2[1] = a2[1][sorted_idxB]

    for i in range(len(a1[1])):
        if a1[1][i] != a2[1][i]:
            dist += (1-float(i)/len(a1[1])) * ( abs(i-np.where(a1[1] == a2[1][i])[0][0])/(len(a1[1])-1) + abs(i - np.where(a2[1] == a1[1][i])[0][0] )/(len(a2[1])-1) )/2
            
    return dist / ((len(a1[0]) + 1 ) / 2)


# PERMUTATION DISTANCE

def count_inversions(arr):
    def merge_count_split_inv(left, right):
        merged, i, j, inv_count = [], 0, 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv_count += len(left) - i  # Tutti gli elementi restanti in left sono inversioni
                j += 1
        merged += left[i:]
        merged += right[j:]
        return merged, inv_count

    def sort_and_count(arr):
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, left_inv = sort_and_count(arr[:mid])
        right, right_inv = sort_and_count(arr[mid:])
        merged, split_inv = merge_count_split_inv(left, right)
        return merged, left_inv + right_inv + split_inv

    _, inv_count = sort_and_count(arr)
    return inv_count

def perm_distance(a1,a2):
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
    num_transpositions = count_inversions(perm)
    
    n = len(seq1)
    max_transpositions = n * (n - 1) // 2  # Numero massimo di swap nel caso peggiore
    return num_transpositions / max_transpositions if max_transpositions > 0 else 0