import deeplabcut
import os
from config_default import *

# os.environ["CUDA_VISIBLE_DEVICES"]="5"
deeplabcut.train_network(config_path, 
                            shuffle=1, 
                            trainingsetindex=0, 
                            gputouse=0, # titan X on rotterdam
                            max_snapshots_to_keep=None) # keep all snapshots 
