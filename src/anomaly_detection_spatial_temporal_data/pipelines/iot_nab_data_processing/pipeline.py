"""
This is a boilerplate pipeline 'iot_nab_data_processing'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from anomaly_detection_spatial_temporal_data.pipelines.iot_gdn_data_processing.nodes import clean_column_names, get_sensor_columns, save_sensor_cols
from .nodes import append_anomaly_column, split_csv_and_save

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
            
            # Append anomaly column NAB
            node(
                func=append_anomaly_column,
                inputs=["params:train_clean_key", "iot_data_train_clean"],
                outputs="iot_data_train_clean_w_label_nab",
                name="append_label_to_train_set1_nab"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:train_anom_key", "iot_data_train_anom_fixed"],
                outputs="iot_data_train_anom_w_label_nab",
                name="append_label_to_train_set2_nab"
            ),
            node(
                func=append_anomaly_column,
                inputs=["params:test_anom_key", "iot_data_test"],
                outputs="iot_data_test_w_label_nab",
                name="append_label_to_test_set_nab"
            ),
            
            # Save splitted CSV
            node(
                func=split_csv_and_save,
                inputs=[
                    "iot_data_train_clean_w_label_nab", 
                    "iot_data_train_anom_w_label_nab", 
                    "iot_data_test_w_label_nab", 
                    "sensor_column_names",
                    "params:nab_ts_process_options_iot"
                ],
                outputs=None,
                name="split_and_save"
            )
            
        ],
        tags="iot_nab_data_processing"
    )
