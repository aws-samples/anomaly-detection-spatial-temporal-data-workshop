"""
This is a boilerplate pipeline 'taddy'
generated using Kedro 0.18.0
"""
import pandas as pd 
import numpy as np 
import os
from typing import Dict, Tuple

from anomaly_detection_spatial_temporal_data.model.data_loader import DynamicDatasetLoader
from anomaly_detection_spatial_temporal_data.model.dynamic_graph import Taddy
from anomaly_detection_spatial_temporal_data.model.model_config import TaddyConfig


def load_data(processed_data, parameters: Dict) -> Dict:
    """
    Load the processed data for the model 
    Args:
        
    Returns: 
        
    """
    data_loader = DynamicDatasetLoader(processed_data, parameters)
    return data_loader.load()

def set_and_train_model(data_dict: Dict, parameters: Dict) -> Tuple:
    """
    Preprocesses the data for transaction history.
    Args:
        financial_data: Raw data.
    Returns:
        Saved edge list, node ids, and transaction labels
    """
    model_config = TaddyConfig(parameters)
    method_obj = Taddy(data_dict, model_config)
    learned_result = method_obj.run()
    return learned_result
    
# def train_model(processed_edge_list: np.ndarray, processed_node_id: pd.DataFrame, parameters: Dict) -> Tuple:
#     """Splits data into training and test sets to be loaded by model.

#     Args:
#         processed_edge_list: Data containing edge list and label.
#         parameters: Parameters defined in parameters/financial_data_processing.yml.
#     Returns:
#         Split data.
#     """
#     financial_fraud_graph = DynamicGraph(processed_edge_list, processed_node_id)
#     rows,cols,labs,weis,headtail,train_size,test_size,n,m = financial_fraud_graph.generateDataset(parameters["train_size"], parameters["snap_size"])
#     return (rows,cols,labs,weis,headtail,train_size,test_size,n,m)
