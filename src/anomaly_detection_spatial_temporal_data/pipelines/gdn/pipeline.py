# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
                inputs=["iot_sensor_cols_txt", "iot_data_gdn_train", "iot_data_gdn_test", "params:train_config", "params:env_config_iot"],
                outputs=None,
                name="train_gdn_iot",
                tags="iot_gdn"
            ),
            node(
                func=set_and_train_model,
                inputs=["wifi_sensor_cols_txt", "wifi_data_gdn_train", "wifi_data_gdn_test", "params:train_config", "params:env_config_wifi"],
                outputs=None,
                name="train_gdn_wifi",
                tags="wifi_gdn"
            )
        ]
    )
