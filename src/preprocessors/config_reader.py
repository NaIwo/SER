from envyaml import EnvYAML


class ConfigReader:
    def __init__(self):
        pass

    @staticmethod
    def read_config(path: str) -> EnvYAML:
        cfg = EnvYAML(path)
        return cfg


config = ConfigReader.read_config('../config.yml')
