"""Project pipelines."""


from typing import Dict

from kedro.pipeline import Pipeline

from anomaly_detection_spatial_temporal_data.pipelines import financial_data_processing as fdp

from anomaly_detection_spatial_temporal_data.pipelines import taddy as model

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """
    financial_data_processing_pipeline = fdp.create_pipeline()
    taddy_model_pipeline = model.create_pipeline()
    return {
        "__default__": financial_data_processing_pipeline + taddy_model_pipeline,
        "financial_fraud_pipeline": financial_data_processing_pipeline+taddy_model_pipeline,
        "dp": financial_data_processing_pipeline,
        "ds": taddy_model_pipeline,
    }

