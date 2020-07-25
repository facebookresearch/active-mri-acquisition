import json

# noinspection PyUnresolvedReferences
import pytest

import actmri.envs


class MockReconstructor:
    def __init__(self, **kwargs):
        self.option1 = kwargs["option1"]
        self.option2 = kwargs["option2"]
        self.option3 = kwargs["option3"]
        self.option4 = kwargs["option4"]


# noinspection PyProtectedMember
class TestActiveMRIEnv:
    mock_config_json_str = """
    {
        "dataset_location": "dummy_location",
        "reconstructor_module": "tests",
        "reconstructor_cls": "MockReconstructor",
        "reconstructor_options": {
            "option1": 1,
            "option2": 0.5,
            "option3": "dummy",
            "option4": true
        },
        "device": "cpu"
    }
    """

    mock_config_dict = json.loads(mock_config_json_str)

    def test_init_from_dict(self):
        env = actmri.envs.ActiveMRIEnv()
        env._init_from_dict(self.mock_config_dict)
        assert type(env._reconstructor) == MockReconstructor
        assert env._reconstructor.option1 == 1
        assert env._reconstructor.option2 == 0.5
        assert env._reconstructor.option3 == "dummy"
        assert env._reconstructor.option4
