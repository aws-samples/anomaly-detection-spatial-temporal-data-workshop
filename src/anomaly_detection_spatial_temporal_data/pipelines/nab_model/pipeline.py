"""
This is a boilerplate pipeline 'nab_model'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import set_and_run_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=set_and_run_model,
                inputs=[ "params:nab_model_options"],
                outputs=None,
                name="run_nab_model_node",
            ),

        ],
        tags="nab"
    )

