import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os


class RealsenseCamera:
    def __init__(self) -> None:
        
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        fps = 30
        width = 640
        height = 480

        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Start streaming
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1 #1 meter
        self.clipping_distance = clipping_distance_in_meters / depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.warm_up()

    def warm_up(self):
        # get the first few frames
        self.get_frame()
        time.sleep(0.01)
        self.get_frame()
        time.sleep(0.01)
        self.get_frame()
        time.sleep(0.01)


    def get_frame(self):
        
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_image_scaled = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_image_scaled, cv2.COLORMAP_JET)

         # the colorizer makes the depth colormap look much better
        colorizer = rs.colorizer()
        depth_colormap2 =  np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data()) 
        
        return color_image, depth_image, depth_colormap2
    
    def stop(self):
        self.pipeline.stop()

if __name__ == '__main__':

    realsense_camera = RealsenseCamera()

    save_folder_name = "./camera_images"
    save_folder = save_folder_name

    folder_counter = 1
    path_exists = os.path.exists(save_folder)
    while(path_exists):
        save_folder = save_folder_name + "_" +  str(folder_counter).zfill(2)
        path_exists = os.path.exists(save_folder)
        folder_counter += 1

    os.makedirs(save_folder)
    img_count = 0

    print("Press t to take  data in and save it. Press ctrl+c to stop the program")

    while True:
        try:
            key = input()
            if key == 't':
                # 1. get image and depth from camera
                colour_img, depth_img, depth_colormap = realsense_camera.get_frame()
                
                for img_name, img in zip(["colour", "depth", "depthmap"], [colour_img, depth_img, depth_colormap]):
                    save_file_path = os.path.join(save_folder, str(img_count).zfill(3) + "." + img_name)
                    if img_name == "depth":
                        np.save(save_file_path + ".npy", img)
                    else:
                        cv2.imwrite(save_file_path + ".png", img)

                img_count += 1

        except KeyboardInterrupt:
            break
        