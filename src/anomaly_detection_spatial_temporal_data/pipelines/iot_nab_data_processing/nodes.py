"""
This is a boilerplate pipeline 'iot_nab_data_processing'
generated using Kedro 0.18.0
"""

import pandas as pd 
import json
from typing import List, Dict
from pathlib import Path
from collections import defaultdict
from anomaly_detection_spatial_temporal_data.pipelines.iot_gdn_data_processing.nodes import ANOMALIES


def append_anomaly_column(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    anomalies = ANOMALIES[dataset_name]
    
    fmt ="%d/%m/%Y %H"
    anomalies_dt = [
        (pd.to_datetime(s, format=fmt), pd.to_datetime(e, format=fmt)) for s, e in anomalies
    ]
    
    df["timestamp"] = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H")
    df = df.set_index(["timestamp"])

    df["attack"] = 0
    for start, end in anomalies_dt:
        df.loc[start:end, "attack"] = 1
        
    return df

def split_csv_and_save(
    train_clean: pd.DataFrame, 
    train_anom: pd.DataFrame,
    test: pd.DataFrame,
    sensor_columns: List[str], 
    parameters: Dict
) -> None:
    csv_save_dir = Path(parameters["ts_data_dir"])
    label_save_dir = Path(parameters["ts_label_dir"])
    csv_save_dir.mkdir(parents=True, exist_ok=True)
    label_save_dir.mkdir(parents=True, exist_ok=True)
    
    combined = pd.concat(
        [
            test            
        ]
    )

    combined_anomalies = ANOMALIES["test_set_anomalies"]
    
    combined_anomalies = [
        (pd.to_datetime(s, format="%d/%m/%Y %H"), pd.to_datetime(e, format="%d/%m/%Y %H")) 
        for s,e in combined_anomalies
    ]
    
    anomaly_dict = defaultdict(list)
    for c in sensor_columns:
        combined.reset_index()[["timestamp", c]].rename(columns={c:"value"}).to_csv(f"{csv_save_dir}/{c}.csv", index=False)

        for s_anom, e_anom in combined_anomalies:
            anomaly_dict[f"{c}.csv"].append([
                s_anom.strftime('%Y-%m-%d %H:%M:%S.%f'), e_anom.strftime('%Y-%m-%d %H:%M:%S.%f')
            ])
    label_dict_filepath = f"{label_save_dir}/labels-combined.json"     
    with open(label_dict_filepath, "w") as fp:
        json.dump(anomaly_dict, fp, indent=4)
    return label_dict_filepath