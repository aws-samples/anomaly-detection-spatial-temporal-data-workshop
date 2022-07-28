# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'iot_gdn_data_processing'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_column_names, get_sensor_columns, append_anomaly_column, normalize, save_sensor_cols, save_csv


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Common
            node(
                func=get_sensor_columns,
                inputs="iot_data_train_clean",
                outputs="sensor_column_names",
                name="get_sensor_column_names_node"
            ),
            node(
                func=clean_column_names,
                inputs="iot_data_train_anom",
                outputs="iot_data_train_anom_fixed",
                name="fix_whitespace_col_name"
            ),
            
            # GDN append
            node(
                func=append_anomaly_column,
                inputs=["params:train_clean_key", "iot_data_train_clean"],
                outputs="iot_data_train_clean_w_label_gdn",
                name="append_label_to_train_set1_gdn"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:train_anom_key", "iot_data_train_anom_fixed"],
                outputs="iot_data_train_anom_w_label_gdn",
                name="append_label_to_train_set2_gdn"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:test_anom_key", "iot_data_test"],
                outputs="iot_data_test_w_label_gdn",
                name="append_label_to_test_set_gdn"
            ),
            
            # GDN normalize
            node(
                func=normalize,
                inputs=["iot_data_train_clean_w_label_gdn", "iot_data_train_anom_w_label_gdn", "iot_data_test_w_label_gdn", "sensor_column_names"],
                outputs=["train_clean_gdn", "train_w_anom_gdn", "test_gdn"],
                name="min-max_normalize_gdn"
            ),
            
            # Common save
            node(
                func=save_sensor_cols,
                inputs="sensor_column_names",
                outputs="iot_sensor_cols_txt",
                name="save_sensor_txt"
            ),
            
            # GDN Save
            node(
                func=save_csv,
                inputs=["train_clean_gdn", "sensor_column_names"],
                outputs="iot_data_gdn_train",
                name="save_train_clean_gdn"
            ),
            node(
                func=save_csv,
                inputs=["train_w_anom_gdn", "sensor_column_names"],
                outputs="iot_data_gdn_train_anom",
                name="save_train_anom_gdn"
            ),
            node(
                func=save_csv,
                inputs=["test_gdn", "sensor_column_names"],
                outputs="iot_data_gdn_test",
                name="iot_data_gdn_test"
            ),
        ]
    )
