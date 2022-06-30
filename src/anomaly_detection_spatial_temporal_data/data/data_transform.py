import time
import pandas as pd
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix,coo_matrix,eye
#from dataset import dataset 
#from utils import ensure_directory
#SAVE_PATH = '../../../../data/02_intermediate/financial_fraud'

class CSVtoTS():
    """Convert raw csv data into format time series models assume"""
    
    def __init__(self, df):
        self.df = df
    
    
class CSVtoStaticGraph():
    """
    Convert raw csv data into format static graph 
    models assume (e.g. node list and edge list)
    """
    
    def __init__(self):
        self.dataset = dataset
    
class CSVtoDynamicGraph():
    """Convert raw csv data into format dynamic graph
    models assume (e.g. node list and edge list with timestamps or sequence index)
    """
    
    def __init__(self, df):
        self.df = df
        
    def _process_edge_data(self, df: pd.DataFrame):
        """
        Get edge list from transaction history to construct graph
        """
        edges = df[['step','customer','merchant','category','amount','fraud']]
        edges_deduped = edges.drop_duplicates(subset=['customer','merchant'], keep='last', )
        edges_array = np.array(edges_deduped[['customer','merchant']])
        vertexs, edges_1d = np.unique(edges_array, return_inverse=True)
        vertex_to_id = {}
        for i,vertex in enumerate(vertexs):
            vertex_to_id.setdefault(vertex,i)
        vertex_to_id_df = pd.DataFrame.from_dict(
            vertex_to_id,  
            orient='index', 
            columns=['idx']
        ).reset_index().rename(columns={"index": "name"})
        edges_idx = np.reshape(edges_1d, [-1, 2])
  
        return vertex_to_id_df, vertex_to_id, edges_deduped

    def _process_label_data(self, edges_deduped: pd.DataFrame, vertex_to_id: Dict):
        """
        Get transaction label and save
        """
        edge_label_arr = np.zeros([edges_deduped.shape[0], 3], dtype=np.int32)
        #for index, row in tqdm(raw_trans_data.iterrows(), total=raw_trans_data.shape[0]):
        for idx, row in tqdm(edges_deduped.reset_index().iterrows(), total=edges_deduped.shape[0]): #using deduped trans 
            #edge_label.setdefault((vertex_to_id[row['customer']],vertex_to_id[row['merchant']]), []).append(row['fraud'])
            edge_label_arr[idx][0] = vertex_to_id[row['customer']]
            edge_label_arr[idx][1] = vertex_to_id[row['merchant']]
            edge_label_arr[idx][2] = row['fraud']
        return edge_label_arr

    def preprocessDataset(self):
        """pre-process dataset"""
        vertex_to_id_df, vertex_to_id, edges_deduped = self._process_edge_data(self.df)
        edge_label_arr = self._process_label_data(edges_deduped, vertex_to_id)
        edge_label_postprocessed_df = pd.DataFrame(edge_label_arr, columns=['source','target','label'])
        return vertex_to_id_df, edge_label_arr, edge_label_postprocessed_df

    
class DynamicGraph():
    """Class to process node and edge list data"""
    def __init__(self, processed_edge_list: np.ndarray, processed_node_id: pd.DataFrame):
        self.edge_list =  processed_edge_list
        self.node_list = processed_node_id
    
    def generateDataset(self, train_per, snap_size):
        m = len(self.edge_list) #edge number 
        n = len(self.node_list) #node number 
        edge_label_arr = self.edge_list
        train_num = int(np.floor(train_per * m))
        train = edge_label_arr[0:train_num, :] #first half being training samples
        test = edge_label_arr[train_num:, :] #second half being test samples 
        t0 = time.time()
        train_mat = csr_matrix(
            (np.ones([np.size(train, 0)], dtype=np.int32), 
             (train[:, 0], train[:, 1])),
            shape=(n, n)
        )
        train_mat = train_mat + train_mat.transpose()
        train_mat = (train_mat + train_mat.transpose() + eye(n)).tolil()
        headtail = train_mat.rows #store the indexes of edges
        degrees = np.array([len(x) for x in headtail])
        
        train_size = int(len(train) / snap_size + 0.5) #making slices of snapshots
        test_size = int(len(test) / snap_size + 0.5)
        print("Train size:%d  %d  Test size:%d %d" %
              (len(train), train_size, len(test), test_size))
        rows = []
        cols = []
        weis = []
        labs = []
        for ii in range(train_size):
            start_loc = ii * snap_size
            end_loc = (ii + 1) * snap_size

            row = np.array(train[start_loc:end_loc, 0], dtype=np.int32)
            col = np.array(train[start_loc:end_loc, 1], dtype=np.int32)
            lab = np.array(train[start_loc:end_loc, 2], dtype=np.int32)
            wei = np.ones_like(row, dtype=np.int32)

            rows.append(row)
            cols.append(col)
            weis.append(wei) #weights
            labs.append(lab) #label
        print("Training dataset contruction finish! Time: %.2f s" % (time.time()-t0))
        t0 = time.time()
        for i in range(test_size):
            start_loc = i * snap_size
            end_loc = (i + 1) * snap_size

            row = np.array(test[start_loc:end_loc, 0], dtype=np.int32)
            col = np.array(test[start_loc:end_loc, 1], dtype=np.int32)
            lab = np.array(test[start_loc:end_loc, 2], dtype=np.int32)
            wei = np.ones_like(row, dtype=np.int32)

            rows.append(row)
            cols.append(col)
            weis.append(wei)
            labs.append(lab)
        print("Test dataset finish constructing! Time: %.2f s" % (time.time()-t0))  
        return rows,cols,labs,weis,headtail,train_size,test_size,n,m