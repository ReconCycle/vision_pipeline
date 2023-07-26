import os
import sys

# ROS
from context_action_framework.types import Module

from labelme_importer import LabelMeImporter
from action_prediction_decision_tree import ActionPredictorDecisionTree


class LLMDataGenerator():
    """
    implement this
    generate data to train LLM
    """

    def __init__(self) -> None:


        self.labelme_importer = LabelMeImporter()
        self.action_prediction_decision_tree = ActionPredictorDecisionTree()
        # load list of images, with ground truths, labelme style
        # convert ground truths to Detections


        datasets_dir = os.path.expanduser("~/datasets2/reconcycle/2023-07-25_disassembly_sequences/")
        
        subfolders = [ f.path for f in os.scandir(datasets_dir) if f.is_dir() ]
        
        print("subfolders", subfolders)
        
        for subfolder in subfolders:
        
            img_paths, all_detections, all_graph_relations, modules = self.labelme_importer.process_labelme_dir(subfolder)

            # TODO: create graph relations
            # TODO: run action_prediction_decision_tree.py

            for img_path, detections, graph_relations, module in zip(img_paths, all_detections, all_graph_relations, modules):
                # print("detections", detections)
                if module is None:
                    print("[red]Module is None!")
                action_type, action_block, reason = self.action_prediction_decision_tree.decision_tree(img_path, detections, graph_relations, module)
                if action_type is not action_type.none:
                    print(f"{reason} -> {module.name}, {action_type.name}")
                else:
                    print(f"[red]{reason} -> {module.name}, {action_type.name}")



if __name__ == '__main__':
    llm_data_generator = LLMDataGenerator()