set -e

env_name="kedro-taddy-venv"

rm -rf ${env_name} 

python -m venv ${env_name}
source ${env_name}/bin/activate

pip install -r requirements_taddy.txt

python -m ipykernel install --user --name ${env_name} --display-name "${env_name}"
