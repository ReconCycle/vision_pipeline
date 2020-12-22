import deeplabcut
import os
import regex as re
from config import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# IMPORTANT: SET THE CONFIG PATH CORRECTLY HERE!
config_path = config_path_work_surface
print("config_path", config_path)
print("deeplabcut.__version__", deeplabcut.__version__)
video_dir = os.path.join(full_path, 'data/work_surface_videos')

videos = [video for video in os.listdir(video_dir) if video.endswith(".avi")]
videos.sort(key=lambda f: int(re.sub('\D', '', f)))
videos = [os.path.join(video_dir, video) for video in videos]
print("videos:", videos)

# create new project
# deeplabcut.create_new_project('work_surface', 'sebastian', videos, copy_videos=False, multianimal=False)

# add more videos if need be later on
# deeplabcut.add_new_videos(config_path, videos, copy_videos=False)

# extract frames
# deeplabcut.extract_frames(config_path,
#                           mode='automatic', algo='all', userfeedback=False, crop=False)

# opens gui for labelling
deeplabcut.label_frames(config_path)

# check the skeleton
# deeplabcut.SkeletonBuilder(config_path)

# deeplabcut.check_labels(config_path)

# crop images for better training
# deeplabcut.cropimagesandlabels(config_path, userfeedback=False)


# deeplabcut.create_training_dataset(config_path)


