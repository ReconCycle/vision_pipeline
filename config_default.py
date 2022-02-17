from pathlib import Path
from config import Config


config_overide_file = Path("config_override.py")
cfg_override = None
if config_overide_file.is_file():
    # override file exists. Import it.
    from config_override import cfg as cfg_override

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
    "camera_calibration_file": "data_limited/camera_calibration/calibration_jsi_f1.3_1450x1450_28-06-2021.yaml",
    "camera_parameters": camera_parameters,

    "dlc_config_file": "data_limited/dlc/work_surface_jsi_05-07-2021/config.yaml",
    
    "yolact_dataset_file": "data_limited/yolact/real_from_synthetic_2021-07-07/dataset.json",
    "yolact_score_threshold": 0.2,
})

if cfg_override is None:
    cfg = main_config.copy()
else:
    print("loaded config from config_override.py")
    cfg = cfg_override


def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]
