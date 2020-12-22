import deeplabcut
import os
from config import *

print("deeplabcut version:", deeplabcut.__version__)
videos = [os.path.join(full_path, "data/kalo_v2_test_videos/test_video_13-11-20.avi")]


# We can export the model if we like
# deeplabcut.export_model(config_path)

# scorername = deeplabcut.analyze_videos(config_path, videos, videotype='.avi')

# plots all individuals. Run this first before tracking individuals
# deeplabcut.create_video_with_all_detections(config_path, videos, scorername)

# deeplabcut.convert_detections2tracklets(config_path, videos, videotype='mp4', shuffle=1, trainingsetindex=0, track_method='skeleton')

# need to look into what this does...
pickle_file = "data/kalo_v2_test_videos/test_video_13-11-20DLC_resnet50_kalo_v2Nov4shuffle1_200000_sk.pickle"
# deeplabcut.refine_tracklets(config_path, pickle_file, videos[0], min_swap_len=2, min_tracklet_len=2, trail_len=50)

# convert pickle to h5
# deeplabcut.convert_raw_tracks_to_h5(config_path, pickle_file)

# filter predictions
# deeplabcut.filterpredictions(config_path, videos, track_method="skeleton")

# deeplabcut.plot_trajectories(config_path, videos, track_method="skeleton")

deeplabcut.create_labeled_video(config_path, videos, draw_skeleton=True, track_method="skeleton")



# follow this guide to work out why things aren't working!!!
# from create_video_with_all_detections we see that we have a few too many detections.
#
# https://github.com/DeepLabCut/DeepLabCut/blob/master/examples/COLAB_maDLC_TrainNetwork_VideoAnalysis.ipynb