# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'iot_ncad_data_processing'
generated using Kedro 0.18.1
"""

import pandas as pd 
from typing import List
from anomaly_detection_spatial_temporal_data.pipelines.iot_gdn_data_processing.nodes import ANOMALIES

def append_anomaly_column(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    anomalies = ANOMALIES[dataset_name]
    
    fmt ="%d/%m/%Y %H"
    anomalies_dt = [
        (pd.to_datetime(s, format=fmt), pd.to_datetime(e, format=fmt)) for s, e in anomalies
    ]
    
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df["pdDateTime"] = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H")
    df = df.set_index(["pdDateTime"])

    df["label"] = 0
    for start, end in anomalies_dt:
        df.loc[start:end, "label"] = 1
        
    return df

def save_csv(
    df: pd.DataFrame, 
    sensor_columns: List[str]
) -> None:
    
    return df[sensor_columns + ["label"]]  
