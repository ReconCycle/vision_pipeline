import deeplabcut
import os
from .config import *
import os.path
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
import pandas as pd
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from deeplabcut.utils import auxiliaryfunctions
import cv2
from skimage.util import img_as_ubyte
from deeplabcut.pose_estimation_tensorflow.predict_videos import checkcropping
from skimage.draw import circle_perimeter, circle, line, line_aa
import time
import platform
import matplotlib as mpl
if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
elif platform.system() == 'Darwin':
    mpl.use('WxAgg') #TkAgg
else:
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import regex as re


class Inference:
    def __init__(self, config_path):
        print("running inference")
        self.DLCscorer = None
        self.DLCscorerlegacy = None
        self.trainFraction = None
        self.cfg = None
        self.dlc_cfg = None
        self.sess = None
        self.inputs = None
        self.outputs = None
        self.pdindex = None

        print("config path", config_path)

        self.initialise_inference(config_path, gputouse=0, batchsize=1)

    def initialise_inference(self, config, shuffle=1, trainingsetindex=0, gputouse=None, batchsize=None,
                             TFGPUinference=True, dynamic=(False, .5, 10)):

        if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
            del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training

        if gputouse is not None: #gpu selection
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

        tf.reset_default_graph()
        start_path=os.getcwd() #record cwd to return to this directory in the end

        cfg = auxiliaryfunctions.read_config(config)
        trainFraction = cfg['TrainingFraction'][trainingsetindex]

        modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
        path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
        try:
            dlc_cfg = load_config(str(path_test_config))
        except FileNotFoundError:
            raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))

        # Check which snapshots are available and sort them by # iterations
        try:
          Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])
        except FileNotFoundError:
          raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))

        if cfg['snapshotindex'] == 'all':
            print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
            snapshotindex = -1
        else:
            snapshotindex=cfg['snapshotindex']

        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]

        print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

        ##################################################
        # Load and setup CNN part detector
        ##################################################

        # Check if data already was generated:
        dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
        trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
        # Update number of output and batchsize
        dlc_cfg['num_outputs'] = cfg.get('num_outputs', dlc_cfg.get('num_outputs', 1))

        if batchsize==None:
            #update batchsize (based on parameters in config.yaml)
            dlc_cfg['batch_size']=cfg['batch_size']
        else:
            dlc_cfg['batch_size']=batchsize
            cfg['batch_size']=batchsize

        if dynamic[0]: #state=true
            #(state,detectiontreshold,margin)=dynamic
            print("Starting analysis in dynamic cropping mode with parameters:", dynamic)
            dlc_cfg['num_outputs']=1
            TFGPUinference=False
            dlc_cfg['batch_size']=1
            print("Switching batchsize to 1, num_outputs (per animal) to 1 and TFGPUinference to False (all these features are not supported in this mode).")

        # Name for scorer:
        DLCscorer,DLCscorerlegacy = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations=trainingsiterations)
        if dlc_cfg['num_outputs']>1:
            if  TFGPUinference:
                print("Switching to numpy-based keypoint extraction code, as multiple point extraction is not supported by TF code currently.")
                TFGPUinference=False
            print("Extracting ", dlc_cfg['num_outputs'], "instances per bodypart")
            xyz_labs_orig = ['x', 'y', 'likelihood']
            suffix = [str(s+1) for s in range(dlc_cfg['num_outputs'])]
            suffix[0] = '' # first one has empty suffix for backwards compatibility
            xyz_labs = [x+s for s in suffix for x in xyz_labs_orig]
        else:
            xyz_labs = ['x', 'y', 'likelihood']

        #sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
        if TFGPUinference:
            sess, inputs, outputs = predict.setup_GPUpose_prediction(dlc_cfg)
        else:
            sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

        pdindex = pd.MultiIndex.from_product([[DLCscorer],
                                              dlc_cfg['all_joints_names'],
                                              xyz_labs],
                                             names=['scorer', 'bodyparts', 'coords'])

        self.DLCscorer = DLCscorer
        self.DLCscorerlegacy = DLCscorerlegacy
        self.trainFraction = trainFraction
        self.cfg = cfg
        self.dlc_cfg = dlc_cfg
        self.sess = sess
        self.inputs = inputs
        self.outputs = outputs
        self.pdindex = pdindex


    # from predict_videos.py, GetPoseS function
    def get_pose(self, img):
        ''' Non batch wise pose estimation for video cap.'''

        nframes = 1

        pose_tensor = predict.extract_GPUprediction(self.outputs, self.dlc_cfg)  # extract_output_tensor(outputs, dlc_cfg)
        PredictedData = np.zeros((nframes, 3 * len(self.dlc_cfg['all_joints_names'])))
        frame = np.array(img)
        frame = img_as_ubyte(frame)

        pose = self.sess.run(pose_tensor, feed_dict={self.inputs: np.expand_dims(frame, axis=0).astype(float)})
        pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]

        PredictedData[0, :] = pose.flatten()  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!

        nframes = 1
        Dataframe = pd.DataFrame(PredictedData, columns=self.pdindex, index=range(nframes))

        bodyparts2plot = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(self.cfg, 'all')

        DLCscorer = self.DLCscorer
        bodyparts2connect = self.cfg['skeleton']

        # recode the bodyparts2connect into indices for df_x and df_y for speed
        bpts2connect = []
        index = np.arange(len(bodyparts2plot))
        for pair in bodyparts2connect:
            if pair[0] in bodyparts2plot and pair[1] in bodyparts2plot:
                bpts2connect.append(
                    [index[pair[0] == np.array(bodyparts2plot)][0], index[pair[1] == np.array(bodyparts2plot)][0]])

        nframes = len(Dataframe.index)

        df_likelihood = np.empty((len(bodyparts2plot), nframes))
        df_x = np.empty((len(bodyparts2plot), nframes))
        df_y = np.empty((len(bodyparts2plot), nframes))
        for bpindex, bp in enumerate(bodyparts2plot):
            df_likelihood[bpindex, :] = Dataframe[DLCscorer, bp, 'likelihood'].values
            df_x[bpindex, :] = Dataframe[DLCscorer, bp, 'x'].values
            df_y[bpindex, :] = Dataframe[DLCscorer, bp, 'y'].values

        return df_x, df_y, df_likelihood, bodyparts2plot, bpts2connect

    def get_labeled_img(self, img, df_x, df_y, df_likelihood, body_parts, bpts2connect, draw_skeleton=True,
                        max_line_distance=300, coords=None, calibration=None, name=""):
        colormap = self.cfg["colormap"]
        dotsize = self.cfg["dotsize"]
        pcutoff = self.cfg["pcutoff"]

        skeleton_color = self.cfg['skeleton_color']

        # calculate skeleton things
        color_for_skeleton = (np.array(mpl.colors.to_rgba(skeleton_color))[:3] * 255).astype(np.uint8)

        colorclass = plt.cm.ScalarMappable(cmap=colormap)
        C = colorclass.to_rgba(np.linspace(0, 1, len(body_parts)))
        colors = (C[:, :3] * 255).astype(np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        ny, nx = img.shape[0], img.shape[1]
        index = 0 # always take the first frame, since nframes = 1
        image = img
        if coords is not None:
            for i in range(len(coords.corners_in_pixels)):
                xc, yc = coords.corners_in_pixels[i]
                xc_meters, yc_m = coords.corners_in_meters[i]
                rr, cc = circle(yc, xc, dotsize, shape=(ny, nx))
                image[rr, cc, :] = colors[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = tuple([int(x) for x in colors[0]])
                cv2.putText(image, coords.corner_labels[i] + ", (" + str(xc_meters) + ", " + str(yc_m) + ")",
                            (xc, yc), font, 1, color, 2, cv2.LINE_AA)

        if len(body_parts) > 4 and calibration is not None:  # don't do this when detecting corners

            # 3D model points in meters, left-handed cartesian coords (x, y, z)
            model_points = np.array([
                (0.0, 0.0, 0.025),  # corner1
                (0.04, 0.0, 0.025),  # corner2
                (0.0, 0.115, 0.025),  # corner3
                (0.04, 0.115, 0.025),  # corner4
                (0.04, 0.0, 0.0),  # corner5
                (0.0, 0.0, 0.0),  # corner6
                (0.04, 0.115, 0.0),  # corner7
                (0.0, 0.115, 0.0),  # corner8
                (0.02, 0.025, 0.025),  # screen_middle
                (0.02, 0.04, 0.025),  # sensor_middle
                (0.02, 0.04, 0.005),  # battery_middle
                (0.02, 0.05525, 0.005),  # object_centre
            ])
            axis = np.float32([[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]]) + model_points[-1]
            mask = np.nonzero(df_likelihood[:, 0] > pcutoff)  # is 0.09
            if len(mask[0]) >= 4:
                image_points = np.hstack((df_x, df_y))
                image_points_masked = image_points[mask]
                model_points_masked = model_points[mask]
                body_parts_masked = np.array(body_parts)[mask]
                camera_matrix, dist_coefs = calibration

                print("body_parts_masked", body_parts_masked)

                # print("camera_matrix", camera_matrix)
                # print("dist_coefs", dist_coefs)

                try:
                    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(model_points_masked, image_points_masked, camera_matrix,
                                                                  dist_coefs)
                    # _, rvecs, tvecs, inliers = cv2.solvePnP(model_points_masked, image_points_masked,
                    #                                               camera_matrix,
                    #                                               dist_coefs,
                    #                                               flags=cv2.SOLVEPNP_EPNP)

                    # project 3D points to image plane
                    projected_axis, _ = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coefs)
                    projected_model_points, _ = cv2.projectPoints(model_points, rvecs, tvecs, camera_matrix, dist_coefs)

                    projected_model_points_masked = projected_model_points[mask]
                    # calculate the error between image_points_masked and projected_model_points_masked

                    pose_estimation_error = np.linalg.norm(np.subtract(image_points_masked, projected_model_points_masked[:, 0, :]))
                    pose_estimation_error = np.mean(pose_estimation_error)
                    print("pose_estimation_error:", pose_estimation_error)

                    name += ", pose error: " + str(int(pose_estimation_error))

                    # only draw 3d model if the fit is good enough
                    if pose_estimation_error < 200:  # TODO: 300 is hardcoded!
                        for p in projected_model_points[:, 0]:
                            cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

                        def draw(img, centre_point, axis_points):
                            the_centre_point = tuple(centre_point.ravel().astype(int))
                            # line is BGR coloured
                            img = cv2.line(img, the_centre_point, tuple(axis_points[0].ravel().astype(int)), (0, 0, 255), 5)
                            img = cv2.line(img, the_centre_point, tuple(axis_points[1].ravel().astype(int)), (0, 255, 0), 5)
                            img = cv2.line(img, the_centre_point, tuple(axis_points[2].ravel().astype(int)), (255, 0, 0), 5)
                            return img

                        image = draw(image, projected_model_points[-1], projected_axis)
                except:
                    # print(ValueError)
                    print("cv2.solvePnPRansac failed!")


        for bpindex in range(len(body_parts)):
            # Draw the skeleton for specific bodyparts to be connected as specified in the config file
            if draw_skeleton:
                for pair in bpts2connect:
                    if (df_likelihood[pair[0], index] > pcutoff) and (df_likelihood[pair[1], index] > pcutoff):
                        # print("pcutoff:", df_likelihood[pair[0], index], df_likelihood[pair[1], index])
                        # print("pair", df_x[pair[0], index], df_y[pair[0], index])
                        a = np.array((df_x[pair[0], index], df_y[pair[0], index]))
                        b = np.array((df_x[pair[1], index], df_y[pair[1], index]))
                        dist = np.linalg.norm(a - b)
                        if dist < max_line_distance:
                            # print("dist", dist)
                            rr, cc, val = line_aa(int(np.clip(df_y[pair[0], index], 0, ny - 1)),
                                                  int(np.clip(df_x[pair[0], index], 0, nx - 1)),
                                                  int(np.clip(df_y[pair[1], index], 1, ny - 1)),
                                                  int(np.clip(df_x[pair[1], index], 1, nx - 1)))
                            image[rr, cc, :] = color_for_skeleton

            if coords is not None:
                x_y_in_meters = coords.coord_transform(np.hstack((df_x, df_y)))
                xc_m = x_y_in_meters[:, 0]
                yc_m = x_y_in_meters[:, 1]

            if df_likelihood[bpindex, index] > pcutoff:
                xc = int(df_x[bpindex, index])
                yc = int(df_y[bpindex, index])
                rr, cc = circle(yc, xc, dotsize, shape=(ny, nx))
                image[rr, cc, :] = colors[bpindex]
                color = tuple([int(x) for x in colors[bpindex]])  # tuple(colors[bpindex])

                if coords is not None:
                    # str(round(df_likelihood[bpindex, index], 2))
                    print_str = body_parts[bpindex] + ", (" \
                                + str(round(xc_m[bpindex], 2)) + ", " + str(round(yc_m[bpindex], 2)) + ")"
                else:
                    print_str = body_parts[bpindex] + ", " + str(round(df_likelihood[bpindex, index], 2))

                cv2.putText(image, print_str, (xc, yc), font, 1.0, color, 2, cv2.LINE_AA)

        if name is not "":
            cv2.putText(image, name, (20, ny - 20), font, 1.0, tuple([int(x) for x in colors[0]]) , 2, cv2.LINE_AA)

        print("")

        return image


    def record_from_webcam(self, mirror=False):
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        frame_count = 0
        cv2.namedWindow("labeled_img", cv2.WINDOW_NORMAL)
        while True:
            t_0 = time.time()
            ret_val, img = cap.read()
            if mirror:
                img = cv2.flip(img, 1)
            # cv2.imshow('my webcam', img)
            df_x, df_y, df_likelihood, bodyparts2plot, bpts2connect = self.get_pose(img)
            labeled_img = self.get_labeled_img(img, df_x, df_y, df_likelihood, bodyparts2plot, bpts2connect)
            cv2.imshow('labeled_img', labeled_img)

            waitkey = cv2.waitKey(1)
            # break
            if waitkey == 27:
                break  # esc to quit
            # elif waitkey == 115:  # "s" key
            #     print("saving image", str(count))
            #     cv2.imwrite(os.path.join(save_images_path, str(count) + '.png'), img)

            t_now = time.time()
            # only start calculating avg fps after the first frame
            if frame_count == 0:
                t_start = time.time()
            else:
                print("avg. FPS:", frame_count / (t_now - t_start))
            print("FPS: ", 1.0 / (t_now - t_0))  # FPS = 1 / time to process loop

            frame_count += 1

        cv2.destroyAllWindows()

    def test_framerate(self):
        # img = np.array(Image.open(an_img))
        img = np.array(cv2.imread(an_img))
        cv2.namedWindow("labeled_img", cv2.WINDOW_NORMAL)
        while True:
            start_time = time.time()
            df_x, df_y, df_likelihood, body_parts, bpts2connect = self.get_pose(img)
            labeled_img = self.get_labeled_img(img, df_x, df_y, df_likelihood, body_parts, bpts2connect)
            cv2.imshow('labeled_img', labeled_img)
            print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop
            waitkey = cv2.waitKey(1)
            # break
            if waitkey == 27:
                break  # esc to quit

    def infer_from_img(self, img_path, coords=None, calibration=None, save_path=None):
        if os.path.isdir(img_path):
            imgs = get_images(img_path)
        else:
            imgs = [img_path]

        cv2.namedWindow("labeled_img", cv2.WINDOW_NORMAL)
        frame_count = 0
        for img_p in imgs:
            img = np.array(cv2.imread(img_p))
            df_x, df_y, df_likelihood, body_parts, bpts2connect = self.get_pose(img)
            labeled_img = self.get_labeled_img(img, df_x, df_y, df_likelihood, body_parts, bpts2connect,
                                               coords=coords, calibration=calibration, name=os.path.basename(img_p))

            cv2.imshow('labeled_img', labeled_img)

            waitkey = cv2.waitKey(1)

            # break
            if waitkey == 27:
                break  # esc to quit
            elif waitkey == ord('p'): # pause
                cv2.waitKey(-1)  # wait until any key is pressed

            t_now = time.time()
            # only start calculating avg fps after the first frame
            if frame_count == 0:
                t_start = time.time()
            else:
                print("avg. FPS:", frame_count / (t_now - t_start))
            frame_count += 1

            if save_path is not None:
                save_file_path = os.path.join(save_path, os.path.basename(img_p))
                print("saving!", save_file_path)
                cv2.imwrite(save_file_path, labeled_img)

            if len(imgs) == 1:
                return df_x, df_y, df_likelihood, body_parts, bpts2connect

    def infer_from_video(self, video_file, should_record=False):

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("total_frames", total_frames)
        print("fps", fps)
        print("width", width)
        print("height", height)

        if should_record:
            resolution = (int(width), int(height))  # (1853, 1543)

            # Define the codec and create VideoWriter object to save the video
            # fourcc = cv2.VideoWriter_fourcc(*'XVID') # very big compression ratio
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_file_dir, video_file_name = os.path.split(video_file)
            video_file_name_split = video_file_name.rsplit('.', 1)
            new_file_path = os.path.join(video_file_dir, video_file_name_split[0] + '_labeled' + '.' + video_file_name_split[1])
            print("new_file_path", new_file_path)
            video_writer = cv2.VideoWriter(new_file_path, fourcc, fps,
                                           resolution)  # native res: 3500, 2900

        count = 0
        cv2.namedWindow("labeled_img", cv2.WINDOW_NORMAL)
        while cap.isOpened():
            ret, frame = cap.read()
            img = np.array(frame)
            if len(img.shape) == 3:
                start_time = time.time()
                df_x, df_y, df_likelihood, body_parts, bpts2connect = self.get_pose(img)
                labeled_img = self.get_labeled_img(img, df_x, df_y, df_likelihood, body_parts, bpts2connect)

                cv2.imshow('labeled_img', labeled_img)
                if should_record:
                    video_writer.write(img)

                print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop
                # cv2.imshow('window-name', frame)
                # cv2.imwrite("frame%d.jpg" % count, frame)
                # time.sleep(3)
                count = count + 1
                waitkey = cv2.waitKey(1)
                # break
                if waitkey == 27:
                    break  # esc to quit
            else:
                break

        cap.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        if should_record:
            cv2.destroyAllWindows()
            video_writer.release()


if __name__ == '__main__':
    # an_img = os.path.join(full_path, "data/kalo_v2/0.png")
    an_img = os.path.join(full_path, "data/kalo_v2_test_imgs_undistorted/8.png")
    # video = os.path.join(full_path, "data/kalo_v2_test_videos/test_video_13-11-20.avi")
    video = os.path.join(full_path, "data/kalo_v2_test_videos/test_video_16-11-2020.avi")
    videos = [video]

    inference = Inference(config_path_kalo)
    # inference.record_from_webcam()
    # inference.test_framerate()
    inference.infer_from_img("data/video_20-11-2020")
    # inference.infer_from_video(video, should_record=True)


