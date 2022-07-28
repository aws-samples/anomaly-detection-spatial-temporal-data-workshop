# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'nab_model'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import set_and_run_model

def create_pipeline(**kwargs) -> Pipeline:

    if kwargs['input_dataset'] == 'iot':
        label_output = "iot_processed_ts_label"
    elif kwargs['input_dataset'] == 'financial':
        label_output = "financial_processed_ts_label"
    else:
        raise NotImplementedError
    return pipeline(
        [
            node(
                func=set_and_run_model,
                inputs=[label_output ,"params:nab_model_options"], #how to switch input?
                outputs=None,
                name="run_nab_model_node",
            ),

        ],
        tags="nab" #add a tag here for choice in kero run
    )

