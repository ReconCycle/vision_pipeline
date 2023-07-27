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
from types import SimpleNamespace


from context_action_framework.types import detections_to_py, gaps_to_py, ros_to_str, str_to_ros
from context_action_framework.types import Action, Detection, Gap, Label, Module, Robot, EndEffector, Camera, Locations
from context_action_framework.srv import NextAction, NextActionResponse
from context_action_framework.msg import CutBlock, LeverBlock, MoveBlock, PushBlock, TurnOverBlock, ViceBlock, VisionBlock
from geometry_msgs.msg import Transform

from graph_relations import GraphRelations
from gap_detection.gap_detector_clustering import GapDetectorClustering

# from helpers import get_action_description

# TODO: implement this
# TODO: taken from action_predictor project. Here we make it much simpler. Given image, predict the action.


class ActionPredictorDecisionTree():
    def __init__(self) -> None:
        config = SimpleNamespace()
        config.realsense = SimpleNamespace()
        config.realsense.debug_clustering = False
            
        self.gap_detector = GapDetectorClustering(config) 
    

    def decision_tree(self, image, detections, graph_relations, module):
        """ TODO: given image, graph_relations, and module, return action
        """
        
        if module == Module.vision:
            dets_hca_front = graph_relations.exists(Label.hca_front)
            dets_hca_side1 = graph_relations.exists(Label.hca_side1)
            dets_hca_side2 = graph_relations.exists(Label.hca_side2)
            
            for dets in [dets_hca_front, dets_hca_side1, dets_hca_side2]:
                if len(dets) > 0:
                    det = dets[0]
                    reason = f"{det.label.name} is visible"
                    return Action.turn_over, TurnOverBlock(Module.vision, det.tf, det.obb_3d, Robot.panda1, EndEffector.soft_hand), reason
            
            dets_hca_back = graph_relations.exists(Label.hca_back)

            if len(dets_hca_back) > 0:
                det_hca_back = dets_hca_back[0]
                if graph_relations.exists(Label.battery):
                    pass

                reason = "hca_back is visible"
                return Action.move, \
                    MoveBlock(Module.vision,
                              det_hca_back.tf,
                              Locations.move_vice_slide.module,
                              Locations.move_vice_slide.tf,
                              det_hca_back.obb_3d,
                              Robot.panda1,
                              EndEffector.soft_hand), reason
            
            
        elif module == Module.vice:
            is_device_in_vice = True # TODO: is device in the vice? 
            if is_device_in_vice:
                # look for clip
                dets_plastic_clip = graph_relations.exists(Label.plastic_clip)
                if len(dets_plastic_clip) > 0:
                    det_plastic_clip = dets_plastic_clip[0]
                    reason = "clip exists"
                    return Action.push, \
                        PushBlock(Locations.push_device_pin_start.module,
                                Locations.push_device_pin_start.tf,
                                Locations.push_device_pin_end.tf,
                                det_plastic_clip.obb_3d,
                                Robot.panda2,
                                EndEffector.screwdriver), reason

                else:
                    # no clip. Try to lever
                    dets_pcb = graph_relations.exists(Label.pcb)
                    dets_pcb_covered = graph_relations.exists(Label.pcb_covered)
                    dets_internals = graph_relations.exists(Label.internals)
                    if len(dets_pcb) > 0 or len(dets_pcb_covered) > 0 or len(dets_internals) > 0:
                        # ! DEBUG ONLY
                        gap = SimpleNamespace()
                        gap.from_tf = None
                        gap.to_tf = None
                        gap.obb_3d = None
                        gaps = [gap] 
                        # ! DEBUG ONLY
                        
                        # TODO: really the gap detection should be only computed when necessary, but maybe not here...
                        
                        # gaps, cluster_img, depth_scaled, device_mask \
                        #     = self.gap_detector.lever_detector(
                        #         self.colour_img,
                        #         self.depth_img,
                        #         detections,
                        #         graph_relations,
                        #         self.camera_info,
                        #         aruco_pose=self.aruco_pose,
                        #         aruco_point=self.aruco_point
                        #     )
                        
                        
                        # TODO: get the actual gaps
                        if gaps is not None and len(gaps) > 0:
                            gap = gaps[0]
                            reason = "[orange]DEBUG gap info needed"
                            return Action.lever, \
                                LeverBlock(Module.vice,
                                        gap.from_tf,
                                        gap.to_tf,
                                        gap.obb_3d,
                                        Robot.panda2,
                                        EndEffector.screwdriver), reason

            else:
                # device not in vice
                # did it fall out, did it land on that sliding trau?
                # TODO: detect device on tray
                # TODO: do we have the graph_relations.not_in relation?
                dets_pcb = graph_relations.exists(Label.pcb)
                dets_pcb_covered = graph_relations.exists(Label.pcb_covered)
                dets_internals = graph_relations.exists(Label.internals)
                dets_battery = graph_relations.exists(Label.battery)
                
                
                # if len(dets_pcb) > 0 and graph_relations.not_in(dets_pcb[0], Label.hca_back) \
                #     and len(dets_pcb_covered) > 0 and graph_relations.not_in(dets_pcb[0], Label.hca_back):  
                lever_success = False
                for dets_component in [dets_pcb, dets_pcb_covered, dets_internals, dets_battery]:
                    if len(dets_component) > 0:
                        det_component = dets_component[0]
                        if graph_relations.not_in(det_component, Label.hca_back):
                            lever_success = True
                        else:
                            lever_success = False
                            reason = f"lever failed, {det_component.label.name} in {Label.hca_back}"
                            return Action.none, None, reason
                
                # find what is on the sliding tray
                det_to_cut = None
                for dets_component in [dets_pcb, dets_pcb_covered, dets_internals]:
                    if len(dets_component) > 0:
                        det_component = dets_component[0]
                        det_to_cut = det_component
                
                if det_to_cut is None:
                    reason = "cannot find det to cut"
                    return Action.none, None, reason
                    
                # if parts on sliding table:
                #   move part with battery to cutter, then cut
                if lever_success and det_to_cut is not None:
                    reason = "internals levered correctly. Now cut."
                    return Action.cut, \
                        CutBlock(Module.vice, ), reason
                

        elif module == Module.cutter:
            
            # TODO: find list of all parts that are on the module and sort them into the finish positions
            
            batteries = graph_relations.exists(Label.battery)
            
            if len(batteries) > 0:
                battery = batteries[0]
                # and if battery is on its own...
                # DONE!
                
                # 
                all_next_to = graph_relations.get_all_next_to(battery)
                all_inside = graph_relations.get_all_inside(battery)
                if len(all_next_to) == 0 and len(all_inside) == 0:
                    # battery is not next to and not inside anything
                    reason = "battery is free"
                    return Action.move, \
                        MoveBlock(Module.vision,
                                battery.tf,
                                Locations.battery_finish.module,
                                Locations.battery_finish.tf,
                                battery.obb_3d,
                                Robot.panda1,
                                EndEffector.soft_hand), reason

                elif len(all_next_to) > 0:
                    # TODO
                    print("[orange]battery is still next to something")
                    pass
                
                elif len(all_inside) > 0:
                    # TODO
                    print("[orange]battery is still inside something")
                    pass
                    


        elif module == Module.cnc:
            pass
            # if device in gripper,
            # if pcb/battery not visible,
            # cut open the device.
            # then cut out the battery tabs
            # then take out battery
            # then take out remaining device
            
        print(f"[red]Module {module.name}: no prediction!")
        return Action.none, None, ""