import json
import os

# noinspection PyUnresolvedReferences
import pytest

import activemri.envs.util


def test_all_configs():
    configs_root = "configs/"
    for fname in os.listdir(configs_root):
        with open(os.path.join(configs_root, fname), "r") as f:
            print(fname)
            data = json.load(f)
            assert "data_location" in data
            assert "device" in data
            assert "reconstructor" in data
            reconstructor_cfg = data["reconstructor"]
            assert "cls" in reconstructor_cfg
            try:
                _ = activemri.envs.util.import_object_from_str(reconstructor_cfg["cls"])
            except ModuleNotFoundError:
                print(f"Reconstructor class in config file {fname} was not found.")
                assert False
            assert "options" in reconstructor_cfg
            assert "checkpoint_path" in reconstructor_cfg
