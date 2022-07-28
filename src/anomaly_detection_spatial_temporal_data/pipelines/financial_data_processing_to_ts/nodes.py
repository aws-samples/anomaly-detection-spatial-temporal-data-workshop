# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'financial_data_processing_to_ts'
generated using Kedro 0.18.0
"""
import pandas as pd 
import json
import numpy as np 
import os
from typing import Dict, Tuple
from pathlib import Path
from collections import defaultdict

from anomaly_detection_spatial_temporal_data.data.data_transform import CSVtoTS

def split_transaction_history_into_time_series(financial_data: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    Preprocesses the data for transaction history.
    Args:
        financial_data: Raw data.
    Returns:
        Saved edge list, node ids, and transaction labels
    """
    financial_fraud_ts = CSVtoTS(financial_data)
    data_csv_save_dir = Path(parameters["ts_data_dir"])
    label_save_dir = Path(parameters["ts_label_dir"])
    data_csv_save_dir.mkdir(parents=True, exist_ok=True)
    label_save_dir.mkdir(parents=True, exist_ok=True)
    
    financial_fraud_ts.split_data_into_separate_ts(
        parameters['ts_count_threshold'], 
        parameters['time_column'], 
        parameters['value_column'], 
        parameters['label_column'], 
        parameters['ts_data_dir'],
        parameters['ts_label_dir'],
    )
    anomaly_dict = defaultdict(list)
    #TODO: add function to form anomaly dict 
    label_dict_filepath = f"{label_save_dir}/labels-combined.json"     
    with open(label_dict_filepath, "w") as fp:
        json.dump(anomaly_dict, fp, indent=4)
    return label_dict_filepath
