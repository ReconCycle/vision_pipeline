import deeplabcut
import os


full_path = "/home/sruiz/projects/reconcycle/deeplabcut"
config_path = os.path.join(full_path, 'kalo1_5-sebastian-2020-10-16/config.yaml')

videos = [os.path.join(full_path, 'videos/cam0.avi'),
          os.path.join(full_path, 'videos/cam1.avi'),
          os.path.join(full_path, 'videos/cam2.avi'),
          os.path.join(full_path, 'videos/cam3.avi'),
          os.path.join(full_path, 'videos/cam4.avi'),
          os.path.join(full_path, 'videos/cam5.avi'),
          os.path.join(full_path, 'videos/cam6.avi'),
          os.path.join(full_path, 'videos/cam7.avi'),
          os.path.join(full_path, 'videos/cam8.avi'),
          os.path.join(full_path, 'videos/cam9.avi'),
          os.path.join(full_path, 'videos/cam10.avi'),
          os.path.join(full_path, 'videos/cam11.avi'),
          os.path.join(full_path, 'videos/cam12.avi'),
          os.path.join(full_path, 'videos/cam13.avi'),
          os.path.join(full_path, 'videos/cam14.avi')]

# create new project
# deeplabcut.create_new_project('kalo1_5', 'sebastian', videos, copy_videos=False, multianimal=False)

# extract frames
# deeplabcut.extract_frames(config_path,
#                           mode='automatic', algo='uniform', crop=False)

# opens gui
# deeplabcut.label_frames(config_path)

# deeplabcut.check_labels(config_path)

# crop images for better training
# deeplabcut.cropimagesandlabels(config_path, userfeedback=False)

# check the skeleton
# deeplabcut.SkeletonBuilder(config_path)

deeplabcut.create_training_dataset(config_path)