"""
This is a boilerplate pipeline 'financial_data_processing_to_ts'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_transaction_history_into_time_series

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_transaction_history_into_time_series,
                inputs=["financial_data", "params:ts_process_options"],
                outputs="financial_processed_ts_label",
                name="split_transaction_data_into_ts_node",
            ),

        ]
    )

