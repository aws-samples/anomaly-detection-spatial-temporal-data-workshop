from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import numpy as np
#from PIL import Image

from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path


class NumpyDataSet(AbstractDataSet):
    """``NumpyDataSet`` loads / save numpy array data from a given filepath as `numpy` array using numpy.

    Example:
    ::

        >>> NumpyDataSet(filepath='/img/file/path.npz')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of ImageDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        #print(protocol)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> np.ndarray:
        """Loads data from the saved file.

        Returns:
            Data from the image file as a numpy array
        """
        #print(self._protocol)
        load_path = get_filepath_str(self._filepath, self._protocol)
        data =  np.load(load_path)
        return data['data']
#         with self._fs.open(load_path, mode="r") as f:
#             data = np.load(f)
#             return data['data']

    def _save(self, data: np.ndarray) -> None:
        """Saves numpy array data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, mode="wb") as f:
            np.savez(f,data=data)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)