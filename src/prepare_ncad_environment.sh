set -e

env_name="kedro-ncad-venv"

rm -rf ${env_name} 

python -m venv ${env_name}
source ${env_name}/bin/activate

pip install -r requirements-ncad.txt
pip install -e anomaly_detection_spatial_temporal_data/model/ncad

pip uninstall -y torchmetrics
pip uninstall -y pytorch-lightning

pip install torchmetrics==0.6.0
pip install pytorch-lightning==1.3.8

python -m ipykernel install --user --name ${env_name} --display-name "${env_name}"