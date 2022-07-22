"""
This is a boilerplate pipeline 'wifi_gdn_data_processing'
generated using Kedro 0.18.1
"""

import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from typing import List, Union, Tuple, Dict
from pathlib import PosixPath

def clean_host_data(udp_subpopulation_data_path: Union[PosixPath, str], filename_prefix_check: str) -> List[pd.DataFrame]:
    all_nums = []
    host_user_map = {}
    maximum_time = get_maximum_time(udp_subpopulation_data_path)
    all_data = []
#     filename_prefix_check = 'r'
#     if sent:
#         filename_prefix_check = 's'
        
    for csv in os.listdir(udp_subpopulation_data_path):
        if csv[0]==filename_prefix_check:
            try: 
                df = pd.read_csv(os.path.join(udp_subpopulation_data_path, csv))
                host_column = df.columns[1]
                host = int(host_column.split("[")[1].split("]")[0])
                df.set_axis(['time', str(host)], axis=1, inplace=True)

                if host not in host_user_map:
                    host_user_map[host] = []
                host_user_map[host].append(csv)
                time_value_map = contruct_time_value_map(df)
                column_values = fill_time_gaps(df, time_value_map, maximum_time)
                df = impute_data(df, maximum_time, column_values)
                all_data.append(df)
            except Exception as e:
                print("cannot load file: ", csv)
                print(e)
                pass
    return all_data

def get_maximum_time(udp_subpopulation_data_path: Union[PosixPath, str]) -> int:
    maximum_time = 0
    for csv in os.listdir(udp_subpopulation_data_path):
        try:
            df = pd.read_csv(os.path.join(udp_subpopulation_data_path, csv))
            df = df.round({"time":0})
            maximum_time = max(max(df["time"]), maximum_time)
        except:
            print("cannot load file:   ", csv)    
    return maximum_time

def contruct_time_value_map(df: pd.DataFrame) -> Dict:
    host = df.columns[1]
    time_value_map = {}
    for index, row in df.iterrows():
        time_value_map[int(row["time"])] = row[host]
    return time_value_map

def fill_time_gaps(df: pd.DataFrame, time_value_map: Dict, maximum_time: int) -> List[str]:
    column_values = []
    for t in range(int(maximum_time)):
        if t+1 in time_value_map:
            column_values.append(time_value_map[t+1])
        else:
            column_values.append(0)
    return column_values

def impute_data(df: pd.DataFrame, maximum_time: int, column_values: List[str]) -> pd.DataFrame:
    time = [i+1for i in range(int(maximum_time))]
    host = df.columns[1]
    df = pd.DataFrame(data={"time": [i+1for i in range(int(maximum_time))], host: column_values})
    return df
            
def construct_dataset_aggregate_hosts(all_host_data: List[pd.DataFrame]) -> pd.DataFrame:
    data = {}
    for df in all_host_data:
        host = df.columns[1]
        if host in data:
            host_data = df[host]
            aggregate_packets = data[host] + df[host]
            data[host] = aggregate_packets
        else:
            data[host] = df[host]

    return pd.DataFrame(data=data)

def subtract_received_from_sent(sent_df:pd.DataFrame, rcvd_df: pd.DataFrame) -> pd.DataFrame:
    sent_minus_received = {}
    for host in sent_df:
        sent_minus_received[host]=sent_df[host] - rcvd_df[host]
    df = pd.DataFrame(data=sent_minus_received)
    return df

def assign_attack_labels(normal_data: pd.DataFrame, anomaly_data: pd.DataFrame) -> Tuple:
    all_data = {}
    for host in normal_data.columns:
        all_data[host] =  abs(normal_data[host]-anomaly_data[host])
    all_data = pd.DataFrame(all_data)
    
    
    all_host_stats = {}
    for host in all_data.columns:
        host_data = dict(all_data[host].describe())
        all_host_stats[host]=host_data
        
    all_host_stats
    anomaly_times = set()
    for host in all_data.columns:
        min_accepted_value = -2*all_host_stats[host]["25%"]
        max_accepted_value =  2*all_host_stats[host]["75%"]
        for index, row in all_data.iterrows():
            value = row[host]
            if (value < min_accepted_value or value > max_accepted_value) and (int(value) != 0):
                anomaly_times.add(index)
                
    anomaly_data["attack"] = [0 for i in range(len(anomaly_data))]
    for index, row in anomaly_data.iterrows():
        if index in anomaly_times:
            anomaly_data["attack"][index]= 1
    return anomaly_data, anomaly_times

def fillna_and_norm(train: pd.DataFrame, test: pd.DataFrame) -> Tuple:
    attack_column = test["attack"]
    train = train.iloc[:, 1:]
    test = test.iloc[:, 1:]
    
    train = train.fillna(train.mean()).fillna(0)
    test = test.fillna(test.mean()).fillna(0)
    
    train_columns = train.columns
    test_columns = test.columns

    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())
    
    test = test.drop(columns=['attack'])

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    train_df = pd.DataFrame(train_ret, columns=train_columns)
    test_df = pd.DataFrame(test_ret, columns=test_columns[:-1])
    test_df["attack"] = attack_column
    
    return train_df, test_df

def create_descriptive_column_names(df: pd.DataFrame) -> Tuple:
    new_column_names = {}
    for column in df.columns:
        if column != "attack":
            new_column_names[column] = "host_"+column
    
    df = df.rename(columns=new_column_names)
    columns_str = "\n".join([c for c in df.columns if c != "attack"])
    return df, columns_str

