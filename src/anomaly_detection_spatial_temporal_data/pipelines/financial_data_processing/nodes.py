# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'financial_data_processing'
generated using Kedro 0.18.0
"""
import pandas as pd 
import numpy as np 
import os
from typing import Dict, Tuple

from anomaly_detection_spatial_temporal_data.data.data_transform import CSVtoDynamicGraph, DynamicGraph

def preprocess_transaction_history(financial_data: pd.DataFrame) -> Tuple:
    """
    Preprocesses the data for transaction history.
    Args:
        financial_data: Raw data.
    Returns:
        Saved edge list, node ids, and transaction labels
    """
    financial_fraud_graph = CSVtoDynamicGraph(financial_data)
    processed_node_id, processed_edge_list, processed_edge_label = financial_fraud_graph.preprocessDataset()
    return processed_node_id, processed_edge_list, processed_edge_label
    
def split_data(processed_edge_list: np.ndarray, processed_node_id: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into training and test sets to be loaded by model.

    Args:
        processed_edge_list: Data containing edge list and label.
        parameters: Parameters defined in parameters/financial_data_processing.yml.
    Returns:
        Split data.
    """
    financial_fraud_graph = DynamicGraph(processed_edge_list, processed_node_id)
    rows,cols,labs,weis,headtail,train_size,test_size,n,m = financial_fraud_graph.generateDataset(parameters["train_size"], parameters["snap_size"])
    return (rows,cols,labs,weis,headtail,train_size,test_size,n,m)

