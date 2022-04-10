import os
from envyaml import EnvYAML


class ConfigReader:
    def __init__(self):
        pass

    @staticmethod
    def read_config(path: str) -> EnvYAML:
        cfg = EnvYAML(path)
        return cfg


source_path = os.path.dirname(os.path.realpath(__file__))
config = ConfigReader.read_config(os.path.join(source_path, 'config.yml'))
