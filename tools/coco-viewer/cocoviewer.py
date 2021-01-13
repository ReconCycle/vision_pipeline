"""Run module
"""
import argparse
from app import App
import os

DATA_DIR = '/home/sruiz/datasets/ndds/13-01-2021-segmented-battery'
JSON_PATH = os.path.join(DATA_DIR, "_coco.json")

parser = argparse.ArgumentParser(description='View images with bboxes from COCO dataset')
parser.add_argument('-i', '--images', default='', type=str, metavar='PATH', help='path to images folder')
parser.add_argument('-a', '--annotations', default='', type=str, metavar='PATH', help='path to annotations json file')


def main():
    args = parser.parse_args()
    if not args.images == "" and not args.annotations == "":
        print("using passed args", args.images, args.annotations)
        app = App(args.images, args.annotations)
    else:
        app = App(DATA_DIR, JSON_PATH)
    app.mainloop()


if __name__ == "__main__":
    main()
