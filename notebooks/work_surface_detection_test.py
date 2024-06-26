# # %%

# import os
# import sys
# import cv2
# import pathlib


# from config import load_config
# from work_surface_detection_opencv import WorkSurfaceDetection

# print(pathlib.Path(__file__).parent.resolve())
# config = load_config(str(pathlib.Path(__file__).parent.resolve() / ".." / "config.yaml"))

# # %%

# img = cv2.imread(os.path.expanduser("~/datasets2/reconcycle/2022-12-05_work_surface/frame0000.jpg"))
# # img = cv2.imread("data_full/dlc/dlc_work_surface_jsi_05-07-2021/labeled-data/raw_work_surface_jsi_08-07-2021/img000.png")

# print("img.shape", img.shape)
# border_width = config.basler.work_surface_ignore_border_width
# print("border_width", border_width)
# # border_width = 100
# # print("border_width", border_width)

# work_surface_det2 = WorkSurfaceDetection(img, border_width, debug=True)
# # %%
