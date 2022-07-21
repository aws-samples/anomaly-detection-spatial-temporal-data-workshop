"""
This is a boilerplate pipeline 'reddit_data_processing'
generated using Kedro 0.18.0
"""
import pandas as pd 
import numpy as np 
import os
from typing import Dict, Tuple
import json

from anomaly_detection_spatial_temporal_data.data.data_transform import CSVtoDynamicGraphWithNodeFeatures

def preprocess_comment_history(comment_history_data_file_path: str) -> Tuple:
    """
    Preprocesses the data for transaction history.
    Args:
        financial_data: Raw data.
    Returns:
        Saved edge list, node ids, and transaction labels
    """
    records = map(json.loads, open(comment_history_data_file_path, encoding="utf8"))
    comment_history_data = pd.DataFrame.from_records(records)
    reddit_history_graph_data = CSVtoDynamicGraphWithNodeFeatures(comment_history_data)
    user_label, u2index, p2index, edgelist_df, data_tvt, user2vec_npy, prod2vec_npy = reddit_history_graph_data.process_data()
    return user_label, u2index, p2index, edgelist_df, data_tvt, user2vec_npy, prod2vec_npy