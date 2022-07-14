set -e

env_name="kedro-gdn-venv"

rm -rf ${env_name} 

python -m venv ${env_name}
source ${env_name}/bin/activate

pip install -r requirements-gdn.txt
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install torch-geometric==1.5.0

python -m ipykernel install --user --name ${env_name} --display-name "${env_name}"