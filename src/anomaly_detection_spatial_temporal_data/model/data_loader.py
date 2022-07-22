"""Reference: https://github.com/yuetan031/TADDY_pytorch"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from numpy.linalg import inv
import pickle
import os
from tqdm import tqdm
from glob import glob
import gc
from collections import defaultdict, Counter
import datetime

import torch
from torch.utils.data import Dataset, DataLoader

class StaticDatasetLoader():
    def __init__(self, dataset_name, args):
        pass

class DynamicGraphWNFDataSet(Dataset):
    """" Sequential Dataset """
    def __init__(self, p2index, item_features, edgelist_df, max_len=30, thresh=0.004):
        self.p2idx = p2index
        self.idx2feats = item_features
        self.edgelist_df = edgelist_df 
        self.tmax, self.tmin = None, None
        self.thresh = thresh
        self.init_data()
        if not max_len:
            self.max_len = self.get_max_len()
        else:
            self.max_len = max_len
        self.uids = list(self.u2pt.keys())

    def __len__(self):
        return len(self.u2pt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        uid = self.uids[idx]
        # Read the data
        data = self.u2pt[uid]
        # Transform the data
        pids, feats, feature_length = self.transform(data)

        # Construct Flags
        repeat_flags = [1 if pid in pids[:idx] else 0 for (idx, pid) in enumerate(pids[1:])]
        target_length = min(len(repeat_flags), self.max_len)
        repeat_flags = repeat_flags + [0] * (self.max_len - target_length) if self.max_len > target_length else repeat_flags[-self.max_len:]
        # Repeat_labels: Binary nd-array. 1-if repeat, 0-if not
        return uid, np.array(feats, dtype=float), np.array(repeat_flags), np.array(feature_length) # torch.FloatTensor(repeat_flags)

    def transform(self, data):
        """
        data: has shape of (n, 2) where dim-0 represents the number of posts,
                    and dim-1 represents pid and time
            The functoin returns 2 lists; first is pids and the second are the features
        """
        pids = []
        feats = []
        data = sorted(data, key=lambda x: x[1])
        for (pid, t) in data:
            pids.append(pid)
            feats.append(self.idx2feats[self.p2idx[pid]])
        # Using zero to pad sequence
        feature_length = min(len(feats), self.max_len)
        feats = feats + [[0]*len(feats[0]) for _ in range(self.max_len-feature_length)] if self.max_len > feature_length else feats[-self.max_len:]

        return pids, np.array(feats), feature_length

    def normalize_t(self, t):
        return (t - self.tmin) / (self.tmax - self.tmin)

    @staticmethod
    def get_p2feats(feats_path):
        gc.disable()

        idx2feats = np.load(open(feats_path + 'post2vec.npy', 'rb'), allow_pickle=True)
        p2idx = pickle.load(open(feats_path + 'post2idx.pkl', 'rb'))

        gc.enable()

        return idx2feats, p2idx

    def get_uids(self, graph_path):
        uids = glob(graph_path)
        return uids

    def init_data(self):
        """initialize dataset from edge list"""
        self.u2pt = defaultdict(list)
        self.total_edges = 0
        for i, row in self.edgelist_df.iterrows():
            self.total_edges += 1
            uid, pid, t = row[0], row[1], int(row[2])
            self.u2pt[uid].append([pid, t])    
#         file = open(self.graph_path)
#         for line in tqdm(file):
#             self.total_edges += 1
#             tmp = line.split(',')
#             uid, pid, t = tmp[0], tmp[1], int(tmp[2])
#             self.u2pt[uid].append([pid, t])
#         file.close()
        remove_ulist=[]
        for u in self.u2pt.keys():
            if len(self.u2pt[u]) < 2: #remove edges with less than 2 interactions
                remove_ulist.append(u)
                self.total_edges -= 1
        for u in remove_ulist:
            del self.u2pt[u]

    def get_max_len(self):
        """get maximum length of user keys"""
        ret = 0
        for u in self.u2pt.keys():
            cur_len = len(self.u2pt[u])
            if cur_len > ret:
                ret = cur_len
        return ret
    
class DynamicGraphWNodeFeatDatasetLoader():
    def __init__(self, labels, u2index, p2index, edge_list, tvt_nids, user_features, item_features):
        self.u2index = u2index
        self.p2index = p2index
        self.edge_list = edge_list #as Pandas DataFrame
        self.tvt_idx = tvt_nids
        self.user_features = user_features
        self.item_features  = item_features
        self.labels = self.process_label(labels)
        self.graph = self.process_edgelist()
        self.print_data_split_info()
    
    def process_label(self, labels_df):
        """process label information"""
        u_all = set()
        pos_uids = set()
        labeled_uids = set()
        labels = np.zeros(len(self.u2index))
        #convert a dataframe to an numpy array, array index being mapped indexes from u2index
        for i,row in labels_df.iterrows():
            author = row['author']
            author_label = row['label']
            u_all.add(author)
            if author_label == 1:
                pos_uids.add(author)
                labeled_uids.add(author)
            elif author_label == 0:
                labeled_uids.add(author)
        print(f'loaded labels, total of {len(pos_uids)} positive users and {len(labeled_uids)} labeled users')
        
        for u in self.u2index:
            if u in pos_uids:
                labels[self.u2index[u]] = 1
        labels = labels.astype(int)
        return labels
    
    def process_edgelist(self):
        """ Load edge list and construct a graph """
        edges = Counter()
        #n = int(graph_num * 10)
        #edgelist_file = f'../data/{ds}/splitted_edgelist_{n}' if n < 10 else f'../data/{ds}/edgelist'

        for i, row in self.edge_list.iterrows():
            u = row[0]
            p = row[1]
            t = row[2]            
            #u = row['author']
            #p = row['subreddit']
            #t = row['retrieved_on']
            edges[(self.u2index[u], self.p2index[p])] += 1
        # Construct the graph
        row = []
        col = []
        entry = []
        for edge, w in edges.items():
            i, j = edge
            row.append(i)
            col.append(j)
            entry.append(w) #save weights (i.e. times of interaction)
        graph = csr_matrix(
            (entry, (row, col)), 
            shape=(len(self.u2index), len(self.p2index))
        )   
        return graph
    
    def print_data_split_info(self):
        idx_train, idx_val, idx_test = self.tvt_idx
        print('Train: total of {:5} users with {:5} pos users and {:5} neg users'.format(
            len(idx_train), 
            np.sum(self.labels[idx_train]), 
            len(idx_train)-np.sum(self.labels[idx_train]))
             )
        print('Val:   total of {:5} users with {:5} pos users and {:5} neg users'.format(
            len(idx_val), 
            np.sum(self.labels[idx_val]), 
            len(idx_val)-np.sum(self.labels[idx_val]))
             )
        print('Test:  total of {:5} users with {:5} pos users and {:5} neg users'.format(
            len(idx_test), 
            np.sum(self.labels[idx_test]), 
            len(idx_test)-np.sum(self.labels[idx_test]))
             )

class DynamicDatasetLoader():

    def __init__(self, processed_data, config):
        self.data = processed_data
        self.c = config['c']
        self.eps = config['eps']
        self.batch_size = config['batch_size']
        self.load_all_tag = config['load_all_tag']
        self.compute_s = config['compute_s']
        self.neighbor_num = config['neighbor_num']
        self.window_size = config['window_size']
        self.eigen_file_name = config['eigen_file_name']
        #self.train_per = args.train_per

    def load_hop_wl_batch(self):  #load the "raw" WL/Hop/Batch dict
        print('Load WL Dictionary')
        f = open('./result/WL/' + self.dataset_name, 'rb')
        wl_dict = pickle.load(f)
        f.close()

        print('Load Hop Distance Dictionary')
        f = open('./result/Hop/hop_' + self.dataset_name + '_' + str(self.neighbor_num) + '_' + str(self.window_size), 'rb')
        hop_dict = pickle.load(f)
        f.close()

        print('Load Subgraph Batches')
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.kneighbor_num) + '_' + str(self.window_size), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        return hop_dict, wl_dict, batch_dict

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix. (0226)"""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation. (0226)"""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation. (0226)"""
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj_np = np.array(adj.todense())
        adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_normalized = self.sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized

    def get_adjs(self, rows, cols, weights, nb_nodes):
        """Generate adjacency matrix and conduct eigenvalue decomposition for node sampling"""
        eigen_file_name = self.eigen_file_name
        if not os.path.exists(eigen_file_name):
            generate_eigen = True
            print('Generating eigen as: ' + eigen_file_name)
        else:
            generate_eigen = False
            print('Loading eigen from: ' + eigen_file_name)
            with open(eigen_file_name, 'rb') as f:
                eigen_adjs_sparse = pickle.load(f)
            eigen_adjs = []
            for eigen_adj_sparse in eigen_adjs_sparse:
                eigen_adjs.append(np.array(eigen_adj_sparse.todense()))

        adjs = []
        if generate_eigen:
            eigen_adjs = []
            eigen_adjs_sparse = []

        for i in range(len(rows)):
            adj = sp.csr_matrix((weights[i], (rows[i], cols[i])), shape=(nb_nodes, nb_nodes), dtype=np.float32)
            adjs.append(self.preprocess_adj(adj))
            if self.compute_s:
                if generate_eigen:
                    eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())
                    for p in range(adj.shape[0]):
                        eigen_adj[p,p] = 0.
                    eigen_adj = self.normalize(eigen_adj)
                    eigen_adjs.append(eigen_adj)
                    eigen_adjs_sparse.append(sp.csr_matrix(eigen_adj))

            else:
                eigen_adjs.append(None)

        if generate_eigen:
            with open(eigen_file_name, 'wb') as f:
                pickle.dump(eigen_adjs_sparse, f, pickle.HIGHEST_PROTOCOL)

        return adjs, eigen_adjs

    def load(self):
        """Load dynamic network dataset"""

        rows, cols, labels, weights, headtail, train_size, test_size, nb_nodes, nb_edges = self.data
    
        degrees = np.array([len(x) for x in headtail])
        num_snap = test_size + train_size

        edges = [np.vstack((rows[i], cols[i])).T for i in range(num_snap)]
        adjs, eigen_adjs = self.get_adjs(rows, cols, weights, nb_nodes)

        labels = [torch.LongTensor(label) for label in labels]

        snap_train = list(range(num_snap))[:train_size]
        snap_test = list(range(num_snap))[train_size:]

        idx = list(range(nb_nodes))
        index_id_map = {i:i for i in idx}
        idx = np.array(idx)

        return {'X': None, 
                'A': adjs, 
                'S': eigen_adjs, 
                'index_id_map': index_id_map, 
                'edges': edges,
                'y': labels, 
                'idx': idx, 
                'snap_train': snap_train, 
                'degrees': degrees,
                'snap_test': snap_test, 
                'num_snap': num_snap}