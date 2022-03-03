import os
import yaml
from helpers import Struct

def load_config(filepath="./config.yaml"):
    if os.path.isfile(filepath):
        with open(filepath, "r") as stream:
            try:
                yaml_parsed = yaml.safe_load(stream)
                return Struct(yaml_parsed)
            except yaml.YAMLError as exc:
                print("yaml error", exc)
    return None

