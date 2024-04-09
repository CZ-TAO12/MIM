import numpy as np
import torch
import pickle
import os 
from torch_geometric.data import Data
import scipy.sparse as ss
from dataLoader import Options, Read_all_cascade

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([col, row])
    data = torch.FloatTensor(coo.data)
    return Data(edge_index=index, edge_attr=data)

def ConRelationGraph(data):
        options = Options(data)
        _u2idx = {} 
    
        with open(options.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        
        edges_list = []
        if os.path.exists(options.net_data):
            with open(options.net_data, 'r') as handle:
                relation_list = handle.read().strip().split("\n")
                relation_list = [edge.split(',') for edge in relation_list]

                relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if edge[0] in _u2idx and edge[1] in _u2idx]
                relation_list_reverse = [edge[::-1] for edge in relation_list]
                edges_list += relation_list_reverse
        else:
            return []

        row, col, entries = [], [], []
        for pair in edges_list:
            row += [pair[0]]
            col += [pair[1]]
            entries += [1.0]
        social_mat = ss.csr_matrix((entries, (row, col)), shape=(len(_u2idx), len(_u2idx)), dtype=np.float32)

        social_matrix = social_mat.dot(social_mat)
        social_matrix = social_matrix.multiply(social_mat) + ss.eye(len(_u2idx), dtype=np.float32)

        social_matrix = social_matrix.tocoo()

        social_matrix = _convert_sp_mat_to_sp_tensor(social_matrix)
        
        return social_matrix

def ConHypergraph(data_name, user_size, window):

    user_size, all_cascade, all_time = Read_all_cascade(data_name)
    user_cont = {}
    for i in range(user_size):
        user_cont[i] = []

    win = window
    for i in range(len(all_cascade)):
        cas = all_cascade[i]

        if len(cas) < win:
            for idx in cas:
                user_cont[idx] = list(set(user_cont[idx] + cas))
            continue
        for j in range(len(cas)-win+1):
            if (j+win) > len(cas):
                break
            cas_win = cas[j:j+win]
            for idx in cas_win:
                user_cont[idx] = list(set(user_cont[idx] + cas_win))

    indptr, indices, data = [], [], []
    indptr.append(0)
    idx = 0

    for j in user_cont.keys():
        if len(user_cont[j])==0:
            idx =  idx + 1
            continue
        source = np.unique(user_cont[j])

        length = len(source)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(source[i])
            data.append(1)
            
    H_U = ss.csr_matrix((data, indices, indptr), shape=(len(user_cont.keys())-idx, user_size))
    HG_User = H_U.tocoo()

    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_cascade)):
        items = np.unique(all_cascade[j])

        length = len(items)

        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(items[i])
            data.append(1)

    H_T = ss.csr_matrix((data, indices, indptr), shape=(len(all_cascade), user_size))
    HG_Item = H_T.tocoo()


    HG_Item = _convert_sp_mat_to_sp_tensor(HG_Item)
    HG_User = _convert_sp_mat_to_sp_tensor(HG_User)

    return HG_Item, HG_User
    