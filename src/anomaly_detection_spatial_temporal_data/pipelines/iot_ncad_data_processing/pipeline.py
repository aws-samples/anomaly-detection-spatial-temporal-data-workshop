"""
This is a boilerplate pipeline 'iot_ncad_data_processing'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from anomaly_detection_spatial_temporal_data.pipelines.iot_gdn_data_processing.nodes import clean_column_names, get_sensor_columns, save_sensor_cols

from .nodes import append_anomaly_column, save_csv

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
            
            # NCAD append
            node(
                func=append_anomaly_column,
                inputs=["params:train_clean_key", "iot_data_train_clean"],
                outputs="iot_data_train_clean_w_label_ncad",
                name="append_label_to_train_set1_ncad"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:train_anom_key", "iot_data_train_anom_fixed"],
                outputs="iot_data_train_anom_w_label_ncad",
                name="append_label_to_train_set2_ncad"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:test_anom_key", "iot_data_test"],
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
            
            # NCAD Save
            node(
                func=save_csv,
                inputs=["iot_data_train_clean_w_label_ncad", "sensor_column_names"],
                outputs="iot_data_ncad_train",
                name="save_train_clean_ncad"
            ),
            node(
                func=save_csv,
                inputs=["iot_data_train_anom_w_label_ncad", "sensor_column_names"],
                outputs="iot_data_ncad_train_anom",
                name="save_train_anom_ncad"
            ),
            node(
                func=save_csv,
                inputs=["iot_data_test_w_label_ncad", "sensor_column_names"],
                outputs="iot_data_ncad_test",
                name="save_test_ncad"
            )
        ]
    )
