"""
This is a boilerplate pipeline 'gdn'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import set_and_train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=set_and_train_model,
                inputs=["iot_sensor_cols_txt", "iot_data_gdn_train", "iot_data_gdn_test", "params:train_config", "params:env_config"],
                outputs=None,
                name="train_gdn"
            )
        ]
    )
