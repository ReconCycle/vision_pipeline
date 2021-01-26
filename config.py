

class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


# ----------------------- CONFIG ----------------------- #


camera_parameters = Config({
    "resize": True,
    "resized_resolution": tuple([1450, 1450]),

    "undistort": True,

    "add_borders": False,
    "camera_new_resolution": tuple([4608, 3288]),
    "camera_offsets": tuple([1000, 100]),  # x-offset, y-offset

    "crop": False,
    "crop_margins": [50, 400, 50, 50], # top, right, bottom, left
})


main_config = Config({
    "camera_calibration_file": "data/camera_calibration/calibration_1450x1450.yaml",
    "camera_parameters": camera_parameters,

    "dlc_config_file": "data/dlc/work_surface-sebastian-2020-11-19/config.yaml",
    "yolact_trained_model": "data/yolact/weights/training_18-01-2021-real/real_266_2400.pth",
    "yolact_config_name": "real", #! not enough, also need to specify the config in yolact/data/config.py
    "yolact_score_threshold": 0.2,
})

cfg = main_config.copy()

def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]

# def set_dataset(dataset_name:str):
#     """ Sets the dataset of the current config. """
#     cfg.dataset = eval(dataset_name)