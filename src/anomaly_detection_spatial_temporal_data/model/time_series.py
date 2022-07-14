"""NAB univariate time series anomaly detector
Prerequisite
------------
run install_nab.sh to install the NAB toolbox first
Examples
--------
>>> model = NABAnomalyDetector()
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


AVAILABLE_MODELS = (
    "ARTime",
    "contextOSE",  # We recommend this one
    "earthgeckoSkyline",
    "relativeEntropy",
)

class TimeSeries():
    def __init__(self):
        pass
    

class NABAnomalyDetector:
    def __init__(self, model_to_use: str, model_path: str, input_path: str, label_path: str, output_path: str):
        assert model_to_use in AVAILABLE_MODELS, (
            f"{model_to_use} is not an available algorithm.\n "
            f"Available models are {AVAILABLE_MODELS}"
        )
        self.model_name = model_to_use
        self.model_path = model_path
        self.input_path = input_path
        self.label_path = label_path
        self.output_path = output_path
        self.nab_path = Path(self.model_path).resolve()
        if not self.nab_path.exists():
            warnings.warn(
                (
                    f"{self.nab_path} does not exist\n"
                    f"please install NAB under {self.nab_path}"
                )
            )


    def _get_absolute_path(self, relative_path: str) -> str:
        """Return the absolute path given path relative to current folder"""
        current_dir = Path(__file__).parent
        return str((current_dir / relative_path).resolve())
    
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

    def predict(self):
        """Predict anomaly records for all csv time series under input dir
        """
        dummy_label_path: str = self._generate_dummy_labels(self.input_path)
        output_dir_path = Path(self.output_path).resolve()
        if output_dir_path.exists():
            shutil.rmtree(output_dir_path)
        output_dir_path.mkdir(exist_ok=True, parents=True)
        command_line = f"""
        cd {self.nab_path} && \
        python3 run.py -d {self.model_name} --detect --skipConfirmation \
        --dataDir {Path(self.input_path).resolve()} \
        --resultsDir {output_dir_path} \
        --windowsFile {Path(self.label_path).resolve()}
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
        print(f"Done. Results saved under {self.output_path}")