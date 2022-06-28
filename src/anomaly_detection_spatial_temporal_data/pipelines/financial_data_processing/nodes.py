"""
This is a boilerplate pipeline 'financial_data_processing'
generated using Kedro 0.18.0
"""
import sys 
sys.path.append('../../')
import pandas as pd 
import numpy as np 
import os

from data.data_transform import CSVtoDynamicGraph
from utils import ensure_directory
SAVE_PATH = '../../../../data/02_intermediate/financial_fraud'

def _process_edge_data(df: pd.DataFrame):
    """
    Get edge list from transaction history to construct graph
    """
    edges = df[['Source','Target']]
    edges = edges.loc[edges.Source!=edges.Target]
    edges = np.array(edges)
    aa, idx = np.unique(edges.tolist(), return_index=True, axis=0) #dedupe 
    edges = edges[np.sort(idx)]
    vertexs, edges_1d = np.unique(edges, return_inverse=True)
    vertex_to_id = {}
    for i,vertex in enumerate(vertexs):
        vertex_to_id.setdefault(vertex,i)
    vertex_to_id_df=pd.DataFrame.from_dict(vertex_to_id,  orient='index', columns=['idx'])
    edges_idx = np.reshape(edges_1d, [-1, 2])
    vertex_to_id_file_save_path = os.path.join(SAVE_PATH, 'vertex_to_id.csv')
    ensure_directory(vertex_to_id_file_save_path)
    vertex_to_id_df.to_csv(vertex_to_id_file_save_path)
    edges_idx_file_save_path = os.path.join(SAVE_PATH, 'edge_list.csv')
    ensure_directory(edges_idx_file_save_path)
    np.savetxt(
        edges_idx_file_save_path
        X=edges_idx,
        delimiter=' ',
        comments='%',
        fmt='%d'
    )
    
def _process_label_data(df: pd.DataFrame):
    """
    Get transaction label 
    """
    edge_label = {}
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        edge_label.setdefault((vertex_to_id[row['Source']],vertex_to_id[row['Target']]), []).append(row['fraud'])
    edge_label_postprocessed = {k: max(edge_label_unique[k]) for k in edge_label_unique}
    #save the labels 
    edge_label_postprocessed_df=pd.DataFrame.from_dict(edge_label_postprocessed,  orient='index', columns=['label'])
    edge_label_file_save_path = os.path.join(SAVE_PATH, 'edge_label.csv')
    ensure_directory(edge_label_file_save_path)
    edge_label_postprocessed_df.to_csv(edge_label_file_save_path)    
    

def preprocess_transaction_history(financial_data: pd.DataFrame):
    """
    Preprocesses the data for transaction history.
    Args:
        financial_data: Raw data.
    Returns:
        Saved edge list, node ids, and transaction labels
    """
    _process_edge_data(financial_data)
    _process_label_data(financial_data)
    
def construct_graph():
    pass
    
    


