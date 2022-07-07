"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

# from anomaly_detection_spatial_temporal_data.pipeline import create_pipeline

from anomaly_detection_spatial_temporal_data.pipelines import iot_data_processing as idp
from anomaly_detection_spatial_temporal_data.pipelines import gdn

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    iot_data_processing_pipeline = idp.create_pipeline()
    gdn_pipeline = gdn.create_pipeline()
    
    return {
        "__default__": iot_data_processing_pipeline,
        "iot_data_processing": iot_data_processing_pipeline,
        "gdn": gdn_pipeline,
        "iot_gdn": iot_data_processing_pipeline + gdn_pipeline
    }
