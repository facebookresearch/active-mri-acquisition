import importlib
import json
import os

# noinspection PyUnresolvedReferences
import pytest


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
            assert "module" in reconstructor_cfg
            try:
                module = importlib.import_module(reconstructor_cfg["module"])
            except ModuleNotFoundError:
                print(f"Reconstructor module in config file {fname} was not found.")
                assert False
            assert "cls" in reconstructor_cfg
            assert getattr(module, reconstructor_cfg["cls"])
            assert "options" in reconstructor_cfg
            assert "checkpoint_path" in reconstructor_cfg
