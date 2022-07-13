"""Project pipelines."""

import os
from typing import Dict

from kedro.pipeline import Pipeline

VENV_INFO = os.environ["VIRTUAL_ENV"]

if "taddy" in VENV_INFO:
    from anomaly_detection_spatial_temporal_data.pipelines import financial_data_processing as fdp
    from anomaly_detection_spatial_temporal_data.pipelines import taddy as dgmodel
    data_processing_pipeline = fdp.create_pipeline()
    model_pipeline = dgmodel.create_pipeline()

if 'nab' in VENV_INFO:
    from anomaly_detection_spatial_temporal_data.pipelines import financial_data_processing_to_ts as fdp_ts
    from anomaly_detection_spatial_temporal_data.pipelines import nab_model 
    data_processing_pipeline = fdp_ts.create_pipeline()
    model_pipeline = nab_model.create_pipeline()

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """

    
    return {
        "__default__": data_processing_pipeline+model_pipeline,
        "data_processing": data_processing_pipeline,
        "model_pipeline": model_pipeline,
    }

