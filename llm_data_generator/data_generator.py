import os
import sys
from rich import print
import natsort

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
        subfolders = natsort.os_sorted(subfolders)
        
        print("subfolders", subfolders)
        
        for subfolder in subfolders:
            
            print(f"\n[green]subfolder: {subfolder}")
        
            img_paths, all_detections, all_graph_relations, modules, cameras = self.labelme_importer.process_labelme_dir(subfolder)
            
            print("===================================================")
            print("===================================================")

            for img_path, detections, graph_relations, module, camera in zip(img_paths, all_detections, all_graph_relations, modules, cameras):
                print(f"img_path: {img_path}")
                print(f"num. dets: {len(detections)}")
                print(f"num. valid dets: {len(graph_relations.valid_detections)}")
                # print("detections", detections)
                if module is None:
                    print("[red]Module is None!")
                action_type, action_block, reason = self.action_prediction_decision_tree.decision_tree(img_path, detections, graph_relations, module)
                
                graph_relations_text = graph_relations.to_text()
                action_text = self.action_to_text(action_type, action_block)
                
                print(graph_relations_text)
                
                if action_type is not action_type.none:
                    print(f"[green]{reason} -> {module.name}, {action_type.name}\n")
                else:
                    print(f"[red]{reason} -> {module.name}, {action_type.name}\n")
                    
                # TODO: generate LLM text.
                
                
                
            print("[red]DEBUG stopped iterating subfolders.")
            break #! DEBUG    

    def action_to_text(self, action_type, action_block):
        # TODO: convert action_type and action_block to text
        
        return "TODO"

if __name__ == '__main__':
    llm_data_generator = LLMDataGenerator()