import os
import yaml

def load_config(filepath="./config.yaml"):
    if os.path.isfile(filepath):
        with open(filepath, "r") as stream:
            try:
                yaml_parsed = yaml.safe_load(stream)
                return Config(yaml_parsed)
            except yaml.YAMLError as exc:
                print("yaml error", exc)
    return None


class Config(object):
    """
    Holds the configuration for anything you want it to.
    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Config(value) if isinstance(value, dict) else value
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)
