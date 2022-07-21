"""Project pipelines. All use cases project pipelines are registered here."""
import os
from typing import Dict

from kedro.pipeline import Pipeline
#from kedro.framework.session import get_current_session
from kedro.framework.session import KedroSession

VENV_INFO = os.environ["VIRTUAL_ENV"]

if "taddy" in VENV_INFO:
    from anomaly_detection_spatial_temporal_data.pipelines import financial_data_processing as fdp
    from anomaly_detection_spatial_temporal_data.pipelines import taddy as dgmodel
    data_processing_pipeline = fdp.create_pipeline()
    model_pipeline = dgmodel.create_pipeline()

if "nab" in VENV_INFO:
    from anomaly_detection_spatial_temporal_data.pipelines import financial_data_processing_to_ts as fdp_ts
    from anomaly_detection_spatial_temporal_data.pipelines import iot_nab_data_processing as idp_ts
    from anomaly_detection_spatial_temporal_data.pipelines import nab_model 
    

    session = KedroSession('anomaly_detection_spatial_temporal_data')
    context = session.load_context()  
    print('Input dataset for NAB model is:',context.params['input_dataset'])
    assert context.params['input_dataset'] in ['iot', 'financial']
    
    if context.params['input_dataset'] == 'iot':
        data_processing_pipeline = idp_ts.create_pipeline() 
    elif context.params['input_dataset'] == 'financial':
        data_processing_pipeline = fdp_ts.create_pipeline()
    else:
        raise NotImplementedError
    #pass down dataset name for NAB model
    model_pipeline = nab_model.create_pipeline(input_dataset=context.params['input_dataset'])

if "gdn" in VENV_INFO:
    from anomaly_detection_spatial_temporal_data.pipelines import iot_gdn_data_processing as idp_gdn
    from anomaly_detection_spatial_temporal_data.pipelines import gdn
    
    data_processing_pipeline = idp_gdn.create_pipeline()
    model_pipeline = gdn.create_pipeline()

if "ncad" in VENV_INFO:
    from anomaly_detection_spatial_temporal_data.pipelines import iot_ncad_data_processing as idp_ncad
    from anomaly_detection_spatial_temporal_data.pipelines import ncad
    
    data_processing_pipeline = idp_ncad.create_pipeline()
    model_pipeline = ncad.create_pipeline()
    
if "eland" in VENV_INFO:
    from anomaly_detection_spatial_temporal_data.pipelines import reddit_data_processing as rdp
    from anomaly_detection_spatial_temporal_data.pipelines import eland
    
    data_processing_pipeline = rdp.create_pipeline()
    model_pipeline = eland.create_pipeline()
    
def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """

    
    return {
        #"__default__": data_processing_pipeline + model_pipeline,
        "data_processing": data_processing_pipeline,
        #"model_pipeline": model_pipeline,
    }

