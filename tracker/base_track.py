from collections import OrderedDict, defaultdict

import numpy as np


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    # since this is used as a class variable, we change count into a dict. 
    _count_dict = defaultdict(int) #! THIS IS USED AS A CLASS VARIABLE

    track_id = 0 #! overriden by STrack instance
    is_activated = False #! overriden by STrack instance
    state = TrackState.New #! overriden by STrack instance

    history = OrderedDict() #? unused
    features = [] #? unused
    curr_feature = None #? unused
    score = 0 #! overriden by STrack instance
    start_frame = 0 #! overriden by STrack instance
    frame_id = 0 #! overriden by STrack instance
    time_since_update = 0 #? unused

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id(cls_id):
        BaseTrack._count_dict[cls_id] += 1
        return BaseTrack._count_dict[cls_id]

    # @even: reset track id
    @staticmethod
    def init_count(num_classes):
        """
        Initiate _count for all object classes
        :param num_classes:
        """
        for cls_id in range(num_classes):
            BaseTrack._count_dict[cls_id] = 0

    @staticmethod
    def reset_track_count(cls_id):
        BaseTrack._count_dict[cls_id] = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed