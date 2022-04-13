import os
from telnetlib import GA
from matplotlib.pyplot import tight_layout
import numpy as np
import time
import cv2
from rich import print
import io

from itertools import permutations
from shapely.geometry import Polygon

from graph_tool.all import *

import matplotlib.pylab as plt
import networkx as nx

from enum import IntEnum
from typing import List
from helpers import Detection


class Action(IntEnum):
    move = 0
    cut = 1
    lever = 2
    turn_over = 3
    remove_clip = 4
    
    
# def compute_is_inside_mask(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2)
#     intersection_area = cv2.countNonZero(intersection)
#     mask1_area = cv2.countNonZero(mask1)
    
#     # stop division by 0
#     if intersection_area < 0.001:
#         return False
    
#     ratio_m1_inside_m2 = intersection_area / mask1_area
    
#     # union = np.logical_or(mask1, mask2)
#     # iou_score = np.sum(intersection) / np.sum(union)
    
#     return ratio_m1_inside_m2 > 0.9

# def compute_is_next_to_mask(mask1, mask2):
#     contours1, _ = cv2.findContours(mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     contours2, _ = cv2.findContours(mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     cnt1 = contours1[0]
#     cnt2 = contours2[0]
    
#     return compute_is_next_to(cnt1, cnt2)

def compute_is_inside(poly1, poly2):
    """ is box1 inside box2? """
    
    intersect = poly1.intersection(poly2).area
    
    # stop division by 0
    if poly1.area < 0.001:
        return False
    
    ratio_p1_inside_p2 = intersect / poly1.area

    
    # print("intersect", intersect)
    # print("ratio_p1_inside_p2", ratio_p1_inside_p2)
    
    # union = p1.union(p2).area
    # iou = intersect / union
    # print("union", union)
    # print("iou", iou)

    # if more than 90% of p1 is inside p2, then it is inside
    return ratio_p1_inside_p2 > 0.9


def compute_is_next_to(poly1, poly2):
    dist = poly1.distance(poly2)
    # print("dist", dist)
    if dist == 0 and (compute_is_inside(poly1, poly2) or compute_is_inside(poly2, poly1)):
        return False

    if dist < 10: # pixels
        return True


class GraphRelations:
    def __init__(self, labels, detections: List[Detection]): 
        self.labels = labels
        self.detections = detections
        
    def using_network_x(self, save_file_path=None):
        labels = self.labels

        inside_edges = []
        next_to_edges = []
        
        # compute polygon for each mask
        # ! I think we do this calculation elsewhere as well. We should only do it once because it is slow
        for detection in self.detections:
            contours, _ = cv2.findContours(np.squeeze(detection.mask).astype(int), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cnt = np.squeeze(contours[0])
            poly = Polygon(cnt)
            detection.mask_poly = poly
        
        for detection1, detection2 in permutations(self.detections, 2): # we use permutations instead because not always associative
            
            # poly1 = Polygon(detection1.obb_corners)
            # poly2 = Polygon(detection2.obb_corners)
            inside = compute_is_inside(detection1.mask_poly, detection2.mask_poly)
            next_to = compute_is_next_to(detection1.mask_poly, detection2.mask_poly)
            
            # inside = compute_is_inside_mask(detection1.mask, detection2.mask)
            # next_to = compute_is_next_to_mask(detection1.mask, detection2.mask)

            if inside:
                # print(detection1.label, "inside", detection2.label)
                inside_edges.append((detection1.id, detection2.id))
            
            if next_to:
                # print(detection1.label, "next to", detection2.label)
                next_to_edges.append((detection1.id, detection2.id))

        edge_labels_dict = {}
        for inside_edge in inside_edges:
            edge_labels_dict[inside_edge] = "in" # inside
            
        for next_to_edge in next_to_edges:
            edge_labels_dict[next_to_edge] = "next to" #nextto

        print("edge_labels_dict", edge_labels_dict)

        node_colors = []
        for detection in self.detections:
            if detection.label == self.labels.battery:
                node_colors.append("blue")
            else:
                node_colors.append("pink")

        G = nx.DiGraph()
        
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure(1, figsize=(12, 9)) # , tight_layout={"pad": 20 }
        fig.clear(True)
        
        G.add_nodes_from([detection.id for detection in self.detections])
        G.add_edges_from(inside_edges + next_to_edges)

        pos = nx.spring_layout(G, k=2)
        
        print("node labels", {id: self.detections[id].label.name for id in G.nodes()})
        
        # add a margin
        ax1 = plt.subplot(111)
        ax1.margins(0.12)
        
        nx.draw(
            G, pos, ax=ax1, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color=node_colors,
            labels={id: self.detections[id].label.name for id in G.nodes()},
            font_size=32
        )

        nx.draw_networkx_edge_labels(G, pos, edge_labels_dict,
                                    label_pos=0.5,
                                    font_size=32
                                    )
        
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.flush()
        io_buf.close()
        
        
        def do_action(label1=None, label2=None, action_type=None):
            sentence = ""
            if action_type == Action.move and label2 is None:
                sentence = "action: move " + label1.name
            elif action_type == Action.cut:
                sentence = "action: cut " + label1.name + "away from" + label2.name
            elif action_type == Action.lever:
                sentence = "action: lever" + label1.name + "away from" + label2.name
            elif action_type == Action.turn_over and label2 is None:
                sentence = "action: turn over" + label1.name
            elif action_type == Action.remove_clip:
                sentence = "action: remove clip" + label1.name + "from" + label2.name
            elif label1 == None and label2 == None and action_type == None:
                sentence = "action: undefined"
            else:
                sentence = "action: unknown"
                
            return label1, label2, action_type, sentence
        
        def is_inside(label1, label2):
            for key, val in edge_labels_dict.items():
                a, b = key
                detection_pair = (self.detections[a].label.value, self.detections[b].label.value)
                if detection_pair == (label1.value, label2.value) and val == "in":
                    return True
                
            return False            
            
        def is_next_to(label1, label2):
            for key, val in edge_labels_dict.items():
                a, b = key
                detection_pair = (self.detections[a].label.value, self.detections[b].label.value)
                if (detection_pair == (label1.value, label2.value) or \
                    detection_pair == (label2.value, label1.value)) and val == "next to":
                    return True
                elif val == "next to":
                    print("detection_pair", detection_pair, (label1.value, label2.value))

            return False
        
        def exists(label):
            for detection in self.detections:
                if detection.label == label:
                    return True
            
            return False
        
        # AI logic
        action = do_action()
        if is_inside(labels.plastic_clip, labels.hca_back):
            action = do_action(labels.plastic_clip, labels.hca_back, Action.remove_clip)
        
        elif is_inside(labels.pcb, labels.hca_back):
            action = do_action(labels.pcb, labels.hca_back, Action.lever)
            
        elif is_inside(labels.pcb_covered, labels.hca_back):
            action = do_action(labels.pcb_covered, labels.hca_back, Action.lever)
        
        elif is_next_to(labels.battery, labels.pcb):
            action = do_action(labels.battery, labels.pcb, Action.cut)    
            
        elif exists(labels.hca_front):
            action = do_action(labels.hca_front, None, Action.turn_over)
        
        return img_arr[:, :, :3], action
    