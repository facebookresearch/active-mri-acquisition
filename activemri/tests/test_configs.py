import json
import os

# noinspection PyUnresolvedReferences
import pytest

import activemri.envs.util


def test_all_configs():
    configs_root = "configs/"
    for fname in os.listdir(configs_root):
        with open(os.path.join(configs_root, fname), "r") as f:
            cfg = json.load(f)
            assert "data_location" in cfg
            assert "device" in cfg
            assert "reward_metric" in cfg
            assert "mask" in cfg
            mask_cfg = cfg["mask"]
            try:
                _ = activemri.envs.util.import_object_from_str(mask_cfg["function"])
            except ModuleNotFoundError:
                print(f"Mask function in config file {fname} was not found.")
                assert False
            assert "args" in mask_cfg and isinstance(mask_cfg["args"], dict)
            assert "reconstructor" in cfg
            reconstructor_cfg = cfg["reconstructor"]
            assert "cls" in reconstructor_cfg
            try:
                _ = activemri.envs.util.import_object_from_str(reconstructor_cfg["cls"])
            except ModuleNotFoundError:
                print(f"Reconstructor class in config file {fname} was not found.")
                assert False
            assert "options" in reconstructor_cfg
            assert "checkpoint_path" in reconstructor_cfg
            assert "transform" in reconstructor_cfg
            try:
                _ = activemri.envs.util.import_object_from_str(
                    reconstructor_cfg["transform"]
                )
            except ModuleNotFoundError:
                print(f"Transform function in config file {fname} was not found.")
                assert False
