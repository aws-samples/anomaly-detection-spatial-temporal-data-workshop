"""Project pipelines."""
import os
from typing import Dict

from kedro.pipeline import Pipeline

VENV_INFO = os.environ["VIRTUAL_ENV"]

if "gdn" in VENV_INFO:
    from anomaly_detection_spatial_temporal_data.pipelines import iot_data_processing as idp
    from anomaly_detection_spatial_temporal_data.pipelines import gdn
    
    data_processing_pipeline = idp.create_pipeline()
    model_pipeline = gdn.create_pipeline()

if "ncad" in VENV_INFO:
    from anomaly_detection_spatial_temporal_data.pipelines import iot_data_processing as idp
    
    data_processing_pipeline = idp.create_pipeline()
    model_pipeline = None

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    return {
        "__default__": data_processing_pipeline,
        "data_processing": data_processing_pipeline,
        "model_pipeline": model_pipeline,
        "complete": data_processing_pipeline + model_pipeline
    }
