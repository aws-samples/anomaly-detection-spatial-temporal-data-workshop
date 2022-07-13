"""
This is a boilerplate pipeline 'iot_gdn_data_processing'
generated using Kedro 0.18.1
"""

import pandas as pd 
import numpy as np 
import os
from typing import Dict, Tuple, List
from sklearn.preprocessing import MinMaxScaler

ANOMALIES = {
    "train_set1_anomalies": [],
    
    # from http://www.batadal.net/images/Attacks_TrainingDataset2.png
    "train_set2_anomalies": [
        ("13/09/2016 23", "16/09/2016 00"),
        ("26/09/2016 11", "27/09/2016 10"),
        ("09/10/2016 09", "11/10/2016 20"),
        ("29/10/2016 19", "02/11/2016 16"),
        ("26/11/2016 17", "29/11/2016 04"),
        ("06/12/2016 07", "10/12/2016 04"),
        ("14/12/2016 15", "19/12/2016 04")
    ],
    
    # http://www.batadal.net/images/Attacks_TestDataset.png
    "test_set_anomalies": [
        ("16/01/2017 09", "19/01/2017 06"),
        ("30/01/2017 08", "02/02/2017 00"),
        ("09/02/2017 03", "10/02/2017 09"),
        ("12/02/2017 01", "13/02/2017 07"),
        ("24/02/2017 05", "28/02/2017 08"),
        ("10/03/2017 14", "13/03/2017 21"),
        ("25/03/2017 20", "27/03/2017 01")
    ]
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def get_sensor_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ["DATETIME", "ATT_FLAG"]]
    
    
def append_anomaly_column(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    anomalies = ANOMALIES[dataset_name]
    
    fmt ="%d/%m/%Y %H"
    anomalies_dt = [
        (pd.to_datetime(s, format=fmt), pd.to_datetime(e, format=fmt)) for s, e in anomalies
    ]
    
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df["pdDateTime"] = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H")
    df = df.set_index(["pdDateTime"])

    df["attack"] = 0
    for start, end in anomalies_dt:
        df.loc[start:end, "attack"] = 1
        
    return df

    
def normalize(
    train_clean: pd.DataFrame,
    train_w_anom: pd.DataFrame,
    test: pd.DataFrame,
    sensor_columns: List[str]
) -> List[pd.DataFrame]:
    
    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train_clean[sensor_columns])
    
    train_clean[sensor_columns] = normalizer.transform(train_clean[sensor_columns])
    train_w_anom[sensor_columns] = normalizer.transform(train_w_anom[sensor_columns])
    test[sensor_columns] = normalizer.transform(test[sensor_columns])
    
    return train_clean, train_w_anom, test
    

def save_sensor_cols(sensor_columns: List[str]) -> None:
    return "\n".join(sensor_columns)

def save_csv(
    df: pd.DataFrame, 
    sensor_columns: List[str], 
#     label_column_name: str,
) -> None:
    
    return df[["timestamp"] + sensor_columns + ["attack"]]
