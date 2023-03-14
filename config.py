import os
import yaml
from helpers import Struct
# import rosparam
from rich import print

def load_config(filepath="./config.yaml"):
    if os.path.isfile(filepath):
        with open(filepath, "r") as stream:
            try:
                
                # param_list = rosparam.load_file(filepath)
                # print("param_list", param_list)
                
                yaml_parsed = yaml.safe_load(stream)
                
                print("config:", yaml_parsed)
                
                return Struct(yaml_parsed)
            except yaml.YAMLError as exc:
                print("yaml error", exc)
    return None

