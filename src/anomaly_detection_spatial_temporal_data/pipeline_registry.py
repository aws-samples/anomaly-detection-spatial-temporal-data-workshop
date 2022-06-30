"""Project pipelines."""


from typing import Dict

from kedro.pipeline import Pipeline

from anomaly_detection_spatial_temporal_data.pipelines import financial_data_processing as dp

from anomaly_detection_spatial_temporal_data.pipelines import taddy as model

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    taddy_model_pipeline = model.create_pipeline()
    return {
        #"__default__": data_processing_pipeline,
        "__default__": data_processing_pipeline + taddy_model_pipeline,
        "dp": data_processing_pipeline,
        "ds": taddy_model_pipeline,
    }

