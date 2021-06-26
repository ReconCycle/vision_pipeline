import deeplabcut
import os
from config_default import *
import pandas as pd

# deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=False)

# NOTE: least bodyparts= something like minimum body parts to consider as an animal
# for multi-animal projects do this for evaluation instead:
deeplabcut.evaluate_multianimal_crossvalidate(config_path, Shuffles=[1], edgewisecondition=True, leastbpts=2, init_points=20, n_iter=50, target='rpck_train')

# other stuff...
# object = pd.read_pickle(os.path.join(full_path, 'kalo_v2-sebastian-2020-11-04/evaluation-results/iteration-0/kalo_v2Nov4-trainset95shuffle1/DLC_resnet50_kalo_v2Nov4shuffle1_200000-snapshot-200000_meta.pickle'))
#
# print(object)
#
# obj = pd.read_hdf(os.path.join(full_path, 'kalo_v2-sebastian-2020-11-11/labeled-data/28/CollectedData_sebastian.h5'))
# print(obj)
# print(obj.columns)