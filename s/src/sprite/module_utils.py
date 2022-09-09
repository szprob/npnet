import os
import pickle
import tempfile
from typing import Dict, List, Optional, Union

import hdfs

from sprite.global_conf import HDFS_ADDRESS, HDFS_PATH, TEMP_PATH


class PreTrainedModule:

    """Pretrained module for all modules.

    Basic class takes care of storing the configuration of the models
    and handles methods for loading ,downloading and saving.

    Attributes:
        pretrained_list  (Optional[List], optional):
            Pretrained model list.
            Defaults to None.
        hdfs_path (str, optional):
            Hdfs path for downloading pretrained models.
            Defaults to HDFS_PATH
    """

    _TEMP_PATH = TEMP_PATH

    def __init__(
        self,
        pretrained_list: Optional[List] = None,
        hdfs_path: str = HDFS_PATH,
    ) -> None:
        # Model temp dir for documents or state dicts
        if not os.path.exists(TEMP_PATH):
            os.mkdir(TEMP_PATH, mode=777)
        self._tmpdir = tempfile.TemporaryDirectory(prefix=f"{TEMP_PATH}/")
        if pretrained_list is None:
            pretrained_list = []
        self._pretrained_list = pretrained_list
        self._hdfs_path = hdfs_path

    def from_pretrained(
        self, model_name: Union[str, Dict], download_path: Optional[str] = None
    ) -> None:
        """Load state dict of `model_name` from hdfs or local path

        Args:
            model_name (Union[str, Dict]): :
                Predtrained model need to be loaded.
                Can be either:
                    - A string, the `model_name` of a pretrained model.
                    - A path to a `directory` containing model weights.
                    - A state dict containing model weights.
            download_path (Optional[str], optional):
                Path the model should be downloaded to.
                If None pretrained model will downloaded to `download_path`.
                Else pretrained model will downloaded to temp path.
                Defaults to None.
        """

        if model_name in self._pretrained_list:
            if download_path is None:
                download_path = self._tmpdir.name
            if os.path.exists(model_name):
                model = model_name
            else:
                self._hdfs_download(
                    local_path=download_path,
                    hdfs_path=f"{self._hdfs_path}/{model_name}.pkl",
                )
            model = f"{self._tmpdir.name}/{model_name}.pkl"
        else:
            model = model_name
        self._load(model)

    def _load(self, model: Union[str, Dict]) -> None:
        """Load  state dict from local model path or dict.

        Args:
            model (Union[str, Dict]):
                Model file need to be loaded.
                Can be either:
                    - A string, the path of a pretrained model.
                    - A state dict containing model weights.
        """

        pass

    def _load_file(self, path: str) -> Dict:
        with open(path, "rb") as f:
            file = pickle.load(f)
        return file

    def _save_file(self, file: Dict, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        with open(path, "wb") as f:
            pickle.dump(file, f)

    def _hdfs_download(
        self, local_path: str, hdfs_path: str, hdfs_address: str = HDFS_ADDRESS
    ) -> None:
        """Download data from hdfs

        Args:
            local_path (str):
                Path for saving local data
            hdfs_path (str):
                Path of hdfs data.
            hdfs_address (str, optional):
                Defaults to HDFS_ADDRESS.
        """
        client = hdfs.InsecureClient(hdfs_address)
        client.download(
            hdfs_path=hdfs_path,
            local_path=local_path,
            overwrite=True,
        )
