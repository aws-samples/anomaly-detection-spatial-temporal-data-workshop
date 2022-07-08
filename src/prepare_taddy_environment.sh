set -e

env_name="kedro-taddy-venv"

rm -rf ${env_name} 

python -m venv ${env_name}
source ${env_name}/bin/activate

pip install -r requirements-taddy.txt
