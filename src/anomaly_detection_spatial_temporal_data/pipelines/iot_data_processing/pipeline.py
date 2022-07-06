"""
This is a boilerplate pipeline 'iot_data_processing'
generated using Kedro 0.18.1
"""

from .nodes import clean_column_names, get_sensor_columns, append_anomaly_column, normalize, save_sensor_cols, save_csv

from kedro.pipeline import Pipeline, node, pipeline
from functools import partial

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
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
            node(
                func=append_anomaly_column,
                inputs=["params:train_clean_key", "iot_data_train_clean", "params:label_col"],
                outputs="iot_data_train_clean_w_label",
                name="append_label_to_train_set1"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:train_anom_key", "iot_data_train_anom_fixed", "params:label_col"],
                outputs="iot_data_train_anom_w_label",
                name="append_label_to_train_set2"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:test_anom_key", "iot_data_test", "params:label_col"],
                outputs="iot_data_test_w_label",
                name="append_label_to_test_set"
            ),
            node(
                func=normalize,
                inputs=["iot_data_train_clean_w_label", "iot_data_train_anom_w_label", "iot_data_test_w_label", "sensor_column_names"],
                outputs=["train_clean", "train_w_anom", "test"],
                name="min-max_normalize"
            ),
            node(
                func=save_sensor_cols,
                inputs="sensor_column_names",
                outputs="iot_gdn_sensor_cols_txt",
                name="save_sensor_txt"
            ),
            node(
                func=save_csv,
                inputs=["train_clean", "sensor_column_names", "params:label_col"],
                outputs="iot_data_gdn_train",
                name="save_train_clean"
            ),
            node(
                func=save_csv,
                inputs=["train_w_anom", "sensor_column_names", "params:label_col"],
                outputs="iot_data_gdn_train_anom",
                name="save_train_anom"
            ),
            node(
                func=save_csv,
                inputs=["test", "sensor_column_names", "params:label_col"],
                outputs="iot_data_gdn_test",
                name="save_test"
            )
        ]
    )
