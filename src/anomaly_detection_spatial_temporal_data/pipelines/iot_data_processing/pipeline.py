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
                inputs=["params:train_clean_key", "iot_data_train_clean", "params:label_column_name_gdn"],
                outputs="iot_data_train_clean_w_label_gdn",
                name="append_label_to_train_set1_gdn"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:train_anom_key", "iot_data_train_anom_fixed", "params:label_column_name_gdn"],
                outputs="iot_data_train_anom_w_label_gdn",
                name="append_label_to_train_set2_gdn"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:test_anom_key", "iot_data_test", "params:label_column_name_gdn"],
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
            
            # NCAD append
            node(
                func=append_anomaly_column,
                inputs=["params:train_clean_key", "iot_data_train_clean", "params:label_column_name_ncad"],
                outputs="iot_data_train_clean_w_label_ncad",
                name="append_label_to_train_set1_ncad"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:train_anom_key", "iot_data_train_anom_fixed", "params:label_column_name_ncad"],
                outputs="iot_data_train_anom_w_label_ncad",
                name="append_label_to_train_set2_ncad"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:test_anom_key", "iot_data_test", "params:label_column_name_ncad"],
                outputs="iot_data_test_w_label_ncad",
                name="append_label_to_test_set_ncad"
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
                inputs=["train_clean_gdn", "sensor_column_names", "params:label_column_name_gdn"],
                outputs="iot_data_gdn_train",
                name="save_train_clean_gdn"
            ),
            node(
                func=save_csv,
                inputs=["train_w_anom_gdn", "sensor_column_names", "params:label_column_name_gdn"],
                outputs="iot_data_gdn_train_anom",
                name="save_train_anom_gdn"
            ),
            node(
                func=save_csv,
                inputs=["test_gdn", "sensor_column_names", "params:label_column_name_gdn"],
                outputs="iot_data_gdn_test",
                name="iot_data_gdn_test"
            ),
            
            # NCAD Save
            node(
                func=save_csv,
                inputs=["iot_data_train_clean_w_label_ncad", "sensor_column_names", "params:label_column_name_ncad"],
                outputs="iot_data_ncad_train",
                name="save_train_clean_ncad"
            ),
            node(
                func=save_csv,
                inputs=["iot_data_train_anom_w_label_ncad", "sensor_column_names", "params:label_column_name_ncad"],
                outputs="iot_data_ncad_train_anom",
                name="save_train_anom_ncad"
            ),
            node(
                func=save_csv,
                inputs=["iot_data_test_w_label_ncad", "sensor_column_names", "params:label_column_name_ncad"],
                outputs="iot_data_ncad_test",
                name="save_test_ncad"
            )
        ]
    )
