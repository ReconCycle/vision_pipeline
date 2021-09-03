import sys
import os 
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import deeplabcut
import regex as re
from config_default import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# IMPORTANT: SET THE CONFIG PATH CORRECTLY HERE!
config_path = "/home/sruiz/projects/reconcycle/vision-pipeline/data_full/dlc/dlc_work_surface_jsi_05-07-2021/config.yaml"
print("config_path", config_path)
print("deeplabcut.__version__", deeplabcut.__version__)
# video_dir = os.path.join(full_path, 'data/work_surface_videos')
# video_dir = "/home/sruiz/projects/reconcycle/vision-pipeline/data/raw_data_work_surface_jsi_05-07-2021"

# videos = [video for video in os.listdir(video_dir) if video.endswith(".avi")]
# videos.sort(key=lambda f: int(re.sub('\D', '', f)))
# videos = [os.path.join(video_dir, video) for video in videos]
# print("videos:", videos)

# create new project
# deeplabcut.create_new_project('dlc_work_surface_jsi_05-07-2021', 'sebastian', videos, copy_videos=False, multianimal=False)

# add more videos if need be later on
# deeplabcut.add_new_videos(config_path, videos, copy_videos=False)

# extract frames
# deeplabcut.extract_frames(config_path,
                        #   mode='automatic', algo='all', userfeedback=False, crop=False)

# opens gui for labelling
# deeplabcut.label_frames(config_path)

# check the skeleton
# deeplabcut.SkeletonBuilder(config_path)

# check the labels
# deeplabcut.check_labels(config_path)

# crop images for better training
# deeplabcut.cropimagesandlabels(config_path, userfeedback=False)


# deeplabcut.create_training_dataset(config_path)

# os.environ["CUDA_VISIBLE_DEVICES"]="5"
# deeplabcut.train_network(config_path, 
#                             shuffle=1, 
#                             trainingsetindex=0, 
#                             # gputouse=0, # titan X on rotterdam
#                             max_snapshots_to_keep=None, # keep all snapshots 
#                             saveiters=1000,
#                             maxiters=20000)

# deeplabcut.evaluate_network(
#     config_path,
#     show_errors=True,
#     gputouse=0,
#     plotting=False
# )


# deeplabcut.analyze_videos(config_path, ['/home/sruiz/projects/reconcycle/vision-pipeline/datasets/raw_data_work_surface_jsi_05-07-2021/0.avi'], save_as_csv=True)

deeplabcut.create_labeled_video(
    config_path, 
    ['/home/sruiz/projects/reconcycle/vision-pipeline/datasets/raw_data_work_surface_jsi_05-07-2021/0.avi'],
    # destfolder='/home/sruiz/projects/reconcycle/vision-pipeline/datasets/raw_data_work_surface_jsi_05-07-2021/output',
    save_frames=False)