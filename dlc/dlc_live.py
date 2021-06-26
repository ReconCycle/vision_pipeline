from dlclive import DLCLive, Processor, benchmark_videos
from config_default import *
from PIL import Image
import numpy as np

an_img = img = np.array(Image.open(os.path.join(full_path, "data/temp2/130.png")))
exported_model_path = os.path.join(full_path, "kalo_v2-sebastian-2020-11-04/exported-models/DLC_kalo_v2_resnet_50_iteration-0_shuffle-1")

dlc_proc = Processor()
dlc_live = DLCLive(exported_model_path, processor=dlc_proc)
dlc_live.init_inference(an_img)
out = dlc_live.get_pose(an_img)
print(out)

videos = [os.path.join(full_path, "data/kalo_v2_videos/10.avi"),
          os.path.join(full_path, "data/kalo_v2_videos/14.avi")]
benchmark_videos(exported_model_path, videos, output=os.path.join(full_path, "data/temp2"), resize=0.5, display=True)
