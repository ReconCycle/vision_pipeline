import os
import sys


# TODO: implement this
# TODO: generate data to train LLM

from labelme_importer import LabelMeImporter

class LLMDataGenerator():
    def __init__(self) -> None:
        pass

        # TODO: load list of images, with ground truths, labelme style, or COCO style?
        # TODO: convert ground truths to Detections


        dataset_dir = os.path.expanduser("~/datasets2/reconcycle/2022-05-02_kalo_qundis/labelme")
        labelme_importer = LabelMeImporter()
        
        img_paths, all_detections = labelme_importer.process_labelme_dir(dataset_dir)

        # TODO: create graph relations
        # TODO: run action_prediction_decision_tree.py

if __name__ == '__main__':
    llm_data_generator = LLMDataGenerator()