
set -e

env_name="kedro-nab-venv"

rm -rf ${env_name} 
python -m venv ${env_name}
source ${env_name}/bin/activate

rm -rf anomaly_detection_spatial_temporal_data/model/NAB
cd anomaly_detection_spatial_temporal_data/model
git clone https://github.com/numenta/NAB.git
rm -fr NAB/data/* NAB/results/*
cd ..
cd .. 

pip install -r requirements_nab.txt
pip install -e anomaly_detection_spatial_temporal_data/model/NAB

python -m ipykernel install --user --name ${env_name} --display-name "${env_name}"


