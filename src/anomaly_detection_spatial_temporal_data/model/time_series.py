"""NAB univariate time series anomaly detector
Prerequisite
------------
run ../../scripts/install_nab.sh to install the NAB toolbox first
Examples
--------
>>> model = IndependentTimeSeriesAnomalyDetector()
>>> model.predict('input_dir', 'output_dir')
"""
import json
import os
import pickle
import shutil
import subprocess
import warnings
from pathlib import Path

import pandas as pd


def _get_absolute_path(relative_path: str) -> str:
    """Return the absolute path given path relative to current folder"""
    current_dir = Path(__file__).parent
    return str((current_dir / relative_path).resolve())


# directory where univariate time series csv files are saved
INPUT_PATH = _get_absolute_path("../../data/timeseries_wholesale/") #wholesale
# directory where anomaly detection results are saved
RESULT_PATH = _get_absolute_path("../../data/detected_anomalies_wholesale/") #wholesale
# where NAB is cloned to
NAB_PATH = _get_absolute_path("../../NAB")
# Default Model. This model generates least false positives
MODEL = "contextOSE"
# change to the filename you created
NODE_NAME_TO_TYPE_MAP_FILE = _get_absolute_path("../../data/processed_wholesale/object_to_type_mapping.pickle") #wholesale

AVAILABLE_MODELS = (
    "ARTime",
    "contextOSE",  # We recommend this one
    "earthgeckoSkyline",
    "relativeEntropy",
)

class TimeSeries():
    def __init__(self):
        pass
    

class NAB(TimeSeries):
    def __init__(self):
        pass    

def install_nab(install_dir: str):
    """Install the Numenta tool box"""
    print(f"Install Numenta under {install_dir}")
    install_script_path = _get_absolute_path("../../scripts/install_nab.sh")
    command = f"""
    cd {Path(install_dir).parent} && \
    bash {install_script_path}
    """
    completed_process = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8"
    )
    print(completed_process.stdout)
    print(completed_process.stderr)


class NABAnomalyDetector:
    def __init__(self, model_to_use: str = "contextOSE"):
        assert model_to_use in AVAILABLE_MODELS, (
            f"{model_to_use} is not an available algorithm.\n "
            f"Available models are {AVAILABLE_MODELS}"
        )
        self.model_name = model_to_use
        self.nab_path = Path(NAB_PATH).resolve()
        if not self.nab_path.exists():
            warnings.warn(
                (
                    f"{self.nab_path} does not exist\n"
                    f"Will install NAB under {self.nab_path}"
                )
            )
            install_nab(str(self.nab_path))
        self.obj_to_type_mapping = None

    def _generate_dummy_labels(self, data_dir: str) -> str:
        """Generate a dummy label JSON file and return its path"""
        data_dir_path = Path(data_dir)
        dummy_labels = dict()
        for file_path in data_dir_path.rglob("*.csv"):
            file_path_relative = file_path.relative_to(data_dir_path)
            dummy_labels[str(file_path_relative)] = []
        dummy_label_path = self.nab_path / "labels" / "dummy_labels.json"
        with dummy_label_path.open("w") as file:
            json.dump(dummy_labels, file, indent=4)
        return str(dummy_label_path.resolve())

    def predict(self, input_data_dir: str, output_dir: str):
        """Predict anomaly records for all csv time series under input dir
        Parameters
        ----------
        input_data_dir : str
            directory where univariate time series csv files are saved
        output_dir : str
            directory where anomaly detection results are saved
        """
        dummy_label_path: str = self._generate_dummy_labels(input_data_dir)
        output_dir_path = Path(output_dir).resolve()
        if output_dir_path.exists():
            shutil.rmtree(output_dir_path)
        output_dir_path.mkdir(exist_ok=True, parents=True)
        command_line = f"""
        cd {self.nab_path} && \
        source activate NAB && \
        python3 run.py -d {self.model_name} --detect --skipConfirmation \
        --dataDir {Path(input_data_dir).resolve()} \
        --resultsDir {output_dir_path} \
        --windowsFile {dummy_label_path}
        """
        print("Predicting ...")
        process = subprocess.run(
            command_line,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
        print(process.stdout)
        print(process.stderr)
        process.check_returncode()
        print(f"Done. Results saved under {output_dir}")

    def _determine_object_type(self, object_name):
        try:
            obj_type = self.obj_to_type_mapping.loc[
                self.obj_to_type_mapping.Object == object_name
                ]["Object_Type"].values[0]
            return obj_type
        except Exception as e:
            print(e)
            print("Did not find such object!")

    def persist_detected_anomalies(
            self,
            object_to_type_mapping=NODE_NAME_TO_TYPE_MAP_FILE,
            result_path = RESULT_PATH,
            #result_folder = RESULT_FOLDER, 
            #data = DATA,
            threshold=0.95
    ):
        """persist detected anomalies to a dict (or DB)"""
        with open(object_to_type_mapping, "rb") as f:
            self.obj_to_type_mapping = pickle.load(f)
        detected_anomalies = {}
        missed_obj = []
        model = self.model_name
        for obj_name in list(self.obj_to_type_mapping.Object):
            obj_type = self._determine_object_type(obj_name)
            if '/' in obj_name:
                obj_name = obj_name.replace('/','-')
            result_filename = f"{model}_{obj_name}.csv"
            result_file = os.path.join(result_path, model, result_filename)
         
            try:
                with open(result_file, "r") as f:
                    ad_result_for_obj = pd.read_csv(f)
                    anomaly_ts = list(
                        ad_result_for_obj.loc[
                            ad_result_for_obj.anomaly_score > threshold
                            ].timestamp
                    )
                    detected_anomalies.setdefault(obj_name, anomaly_ts)
            except Exception as e:
                print(e)
                missed_obj.append(obj_name)

        detected_anomalies_file_path = os.path.join(
            result_path, model, f"detected_anomalies.pickle"
        )            
        pickle.dump(detected_anomalies, open(detected_anomalies_file_path, "wb"), -1)
        return detected_anomalies, missed_obj