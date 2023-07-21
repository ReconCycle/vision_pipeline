from argparse import Action
import os
import numpy as np
import time
import cv2
from rich import print
import json
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import rospy
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
from std_msgs.msg import String
import message_filters
from rospy_message_converter import message_converter

from context_action_framework.types import detections_to_py, gaps_to_py, ros_to_str, str_to_ros
from context_action_framework.types import Action, Detection, Gap, Label, Module, Robot, EndEffector, Camera, Locations
from context_action_framework.srv import NextAction, NextActionResponse
from context_action_framework.msg import CutBlock, LeverBlock, MoveBlock, PushBlock, TurnOverBlock, ViceBlock, VisionBlock
from geometry_msgs.msg import Transform

from graph_relations import GraphRelations

# from helpers import get_action_description

# TODO: implement this
# TODO: taken from action_predictor project. Here we make it much simpler. Given image, predict the action.


class ActionPredictorDecisionTree():
    def __init__(self) -> None:
        pass
    

    def decision_tree(self, image, detections, graph_relations, module):
        """ TODO: given image, graph_relations, and module, return action
        """
        
        if module == Module.vision:
        
            if graph_relations.exists(Label.hca_front):
                return Action.turn_over, TurnOverBlock(Module.vision, det.tf, det.obb_3d, Robot.panda1, EndEffector.soft_hand)

            if graph_relations.exists(Label.hca_back):
                if graph_relations.exists(Label.battery):
                    pass
                    print("battery is visible")

                return Action.move, \
                    MoveBlock(Module.vision,
                              det_hca_back.tf,
                              Locations.move_vice_slide.module,
                              Locations.move_vice_slide.tf,
                              det_hca_back.obb_3d,
                              Robot.panda1,
                              EndEffector.soft_hand)

            
            
        elif module == Module.vice:
            is_device_in_vice = True # TODO: is device in the vice? 
            if is_device_in_vice:
                # look for clip
                dets_plastic_clip = graph_relations.exists(Label.plastic_clip)
                if len(dets_plastic_clip) > 0:
                    det_plastic_clip = dets_plastic_clip[0]
                    return success, pos, Action.push, \
                        PushBlock(Locations.push_device_pin_start.module,
                                Locations.push_device_pin_start.tf,
                                Locations.push_device_pin_end.tf,
                                det_plastic_clip.obb_3d,
                                Robot.panda2,
                                EndEffector.screwdriver)

                else:
                    # no clip. Try to lever
                    dets_pcb = graph_relations.exists(Label.pcb)
                    dets_pcb_covered = graph_relations.exists(Label.pcb_covered)
                    dets_internals = graph_relations.exists(Label.internals)
                    if len(dets_pcb) > 0 or len(dets_pcb_covered) > 0 or len(dets_internals) > 0:
                        if gaps is not None and len(gaps) > 0:
                            gap = gaps[0]
                            pos += 1
                            return success, pos, Action.lever, \
                                LeverBlock(Module.vice,
                                        gap.from_tf,
                                        gap.to_tf,
                                        gap.obb_3d,
                                        Robot.panda2,
                                        EndEffector.screwdriver)

            else:
                # did it fall out, did it land on that sliding table?
                pass
                # if parts on sliding table:
                #   move part with battery to cutter, then cut 

        elif module == Module.cutter:
            if graph_relations.exists(Label.battery):
                # and if battery is on its own...
                # DONE!
                pass

        elif module == Module.cnc:
            pass
            # if device in gripper,
            # if pcb/battery not visible,
            # cut open the device.
            # then cut out the battery tabs
            # then take out battery
            # then take out remaining device
    