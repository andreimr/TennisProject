import cv2
from court_detection_net import CourtDetectorNet
import numpy as np
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect
from homography import is_pt_inside_court
from datetime import datetime
import os
import pandas as pd
import argparse
import torch


def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps


def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img


def get_ground_truth(video_file, args):
    file_gt = video_file.replace('.mp4', '_out.txt')
    with open(file_gt, 'r') as file:
        lines = file.readlines()
        bounce_list = []
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line[0]
            bounce_list.append(int(parts))
    return bounce_list


def evaluate(bounce_list, bounce_gt_list, args):
    num_tp, num_fp, num_tn, num_fn = 0, 0, 0, 0
    FP_frames, FN_frames = [], []

    if len(bounce_list) != len(bounce_gt_list):
        print('Error: lengths of bounce list and ground truth list are different')
        return num_tp, num_fp, num_tn, num_fn, [], []
    
    for i in range(min(len(bounce_list), len(bounce_gt_list))):
        if bounce_list[i] == bounce_gt_list[i]:
            if bounce_list[i] == 1:
                num_tp += 1
            else:
                num_tn += 1
        else:
            if bounce_list[i] == 1:
                num_fp += 1
                FP_frames.append(i)
            else:
                num_fn += 1
                FN_frames.append(i)

    return num_tp, num_fp, num_tn, num_fn, FP_frames, FN_frames


# MODIFICATION
def run_detection(frames, scenes, bounces, ball_track, homography_matrices, kps_court, 
         draw_trace=False, trace=7, reject_bounces_outside_court=False):
    """
    :params
        frames: list of original images
        scenes: list of beginning and ending of video fragment
        bounces: list of image numbers where ball touches the ground
        ball_track: list of (x,y) ball coordinates
        homography_matrices: list of homography matrices
        kps_court: list of 14 key points of tennis court
        persons_top: list of person bboxes located in the top of tennis court
        persons_bottom: list of person bboxes located in the bottom of tennis court
        draw_trace: whether to draw ball trace
        trace: the length of ball trace
    :return
        imgs_res: list of resulting images, 
        filtered_bounces: set of bounces filtered by this algorithm
    """
    imgs_res = []
    width_minimap = 166
    height_minimap = 350

    # create empty set for filtered bounces
    filtered_bounces = set()

    is_track = [x is not None for x in homography_matrices] 
    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]

        eps = 1e-15
        scene_rate = sum_track/(len_track+eps)
        if (scene_rate > 0.5):
            court_img = get_court_img()

            for i in range(scenes[num_scene][0], scenes[num_scene][1]):
                img_res = frames[i]
                inv_mat = homography_matrices[i]

                # draw ball trajectory
                if ball_track[i][0]:
                    if draw_trace:
                        for j in range(0, trace):
                            if i-j >= 0:
                                if ball_track[i-j][0]:
                                    draw_x = int(ball_track[i-j][0])
                                    draw_y = int(ball_track[i-j][1])
                                    img_res = cv2.circle(frames[i], (draw_x, draw_y),
                                    radius=3, color=(0, 255, 0), thickness=2)
                    else:    
                        img_res = cv2.circle(img_res , (int(ball_track[i][0]), int(ball_track[i][1])), radius=5,
                                             color=(0, 255, 0), thickness=2)
                        img_res = cv2.putText(img_res, 'ball', 
                              org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.8,
                              thickness=2,
                              color=(0, 255, 0))

                # draw court keypoints
                if kps_court[i] is not None:
                    for j in range(len(kps_court[i])):
                        img_res = cv2.circle(img_res, (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                          radius=0, color=(0, 0, 255), thickness=10)

                height, width, _ = img_res.shape

                # draw bounce in minimap if bounce is detected inside the court.
                if i in bounces and inv_mat is not None:
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)

                    # MODIFICATION
                    # check if the bounce is inside the court
                    if not reject_bounces_outside_court or is_pt_inside_court(ball_point[0, 0]):
                        # draw bounce in the minimap
                        court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                        radius=0, color=(0, 255, 255), thickness=50)
                        # add this frame ID to the set of filtered bounces
                        filtered_bounces.add(i)

                # (removed person rendering)

                minimap = court_img.copy()
                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
                imgs_res.append(img_res)

        else:    
            imgs_res = imgs_res + frames[scenes[num_scene][0]:scenes[num_scene][1]] 
    return imgs_res, filtered_bounces        
 

def write(imgs_res, fps, path_output_video):
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(imgs_res)):
        frame = imgs_res[num]
        out.write(frame)
    out.release()    




def run_trial(path_to_video_file, video_file, ball_detector, bounce_detector, court_detector, reject_bounces_outside_court):
    video_filepath = path_to_video_file + '/' + video_file

    # Read the video file
    frames, fps = read_video(video_filepath)
    scenes = scene_detect(video_filepath)    

    print('ball detection')
    ball_track = ball_detector.infer_model(frames, video_file=video_filepath.split('/')[-1])

    print('court detection')
    homography_matrices, kps_court = court_detector.infer_model(frames)

#   No utility for this prototype
#    print('person detection')
#    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

    # bounce detection
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    imgs_res, filtered_bounces = run_detection(frames, scenes, bounces, ball_track, homography_matrices, kps_court,
                    draw_trace=True, reject_bounces_outside_court=reject_bounces_outside_court)

    # Generate a list corresponding to each frame, where 1 (0) represents the presence (absence) of a bounce in the frame    
    bounce_list = [1 if i in filtered_bounces else 0 for i in range(len(frames))]
    return imgs_res, fps, bounce_list


def main(args):
    # debug mode?
    debug_mode = args.debug
    if debug_mode:
        print('Running in debug mode')

    # algorithm parameters

    # ball tracking parameters
    binary_threshold = 127
    hough_min_dist = 1
    hough_param1 = 50
    hough_param2 = 2
    hough_min_radius = 2 
    hough_max_radius = 7

    # bounce detection parameters
    bounce_threshold = 0.45
    distance_threshold = 80
    reject_bounces_outside_court = True


    # open the INI file containing the parameters, and update each parameter if it is present in the file
    if args.param_file:
        with open(args.param_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                # remove everything in `line' after the first comment character
                line = line.split('#')[0]
                line = line.split(';')[0]
                line = line.split('[')[0]
                if len(line) == 0:
                    continue
                parts = line.split('=')
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                value = parts[1].strip()
                if key == 'binary_threshold':
                    binary_threshold = int(value)
                elif key == 'hough_min_dist':
                    hough_min_dist = int(value)
                elif key == 'hough_param1':
                    hough_param1 = int(value)
                elif key == 'hough_param2':
                    hough_param2 = int(value)
                elif key == 'hough_min_radius':
                    hough_min_radius = int(value)
                elif key == 'hough_max_radius':
                    hough_max_radius = int(value) 
                elif key == 'bounce_threshold':
                    bounce_threshold = float(value)
                elif key == 'distance_threshold':
                    distance_threshold = int(value)
                elif key == 'reject_bounces_outside_court':
                    reject_bounces_outside_court = bool(value)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running models on device: {}".format(device))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_output_folder = args.path_output_folder + '/' + timestamp + '/'
    print('output folder: {}'.format(path_output_folder))
    os.mkdir(path_output_folder)

    print('load ball tracking model {}'.format(args.path_ball_track_model))
    ball_detector = BallDetector(args.path_ball_track_model, 
                                 device, 
                                 distance_threshold,
                                 binary_threshold, 
                                 hough_min_dist, 
                                 hough_param1, 
                                 hough_param2, 
                                 hough_min_radius, 
                                 hough_max_radius,
                                 debug_mode,
                                 path_output_folder)
    print('load bounce detection model {}'.format(args.path_bounce_model))
    bounce_detector = BounceDetector(args.path_bounce_model,
                                     bounce_threshold,
                                     distance_threshold)
    
    print('load court detection model {}'.format(args.path_court_model))
    court_detector = CourtDetectorNet(args.path_court_model, 
                                      device)

    # Create a list to store the results
    results = []

    sum_tp, sum_fp, sum_tn, sum_fn = 0, 0, 0, 0
    FP_frames, FN_frames = [], []

    # check to see if the file exists
    if not os.path.exists(args.file_input_video_list):
        print('Video file list {} does not exist'.format(args.file_input_video_list))
        return

    with open(args.file_input_video_list, 'r') as file_list:
        lines = file_list.readlines()
        for line in lines:
            video_file = line.strip()
            if video_file.startswith('#'):
                continue
            if video_file.endswith('.mp4'):

                ## RUN TRIAL
                print("Processing video file: {}".format(video_file))
                imgs_res, fps, bounce_list = run_trial(args.path_input_video_folder, video_file, 
                                            ball_detector, 
                                            bounce_detector,
                                            court_detector,
                                            reject_bounces_outside_court=reject_bounces_outside_court)
                
                ## EVALUATE RESULT
                bounce_gt_list = get_ground_truth(args.path_input_video_folder + '/' + video_file, args)
                num_tp, num_fp, num_tn, num_fn, FP_frames, FN_frames = evaluate(bounce_list, bounce_gt_list, args)
                sum_tp += num_tp
                sum_fp += num_fp
                sum_tn += num_tn
                sum_fn += num_fn

                # write the resulting images to a video file
                output_video_file = path_output_folder + video_file.replace('.mp4', '_output.mp4')
                write(imgs_res, fps, output_video_file)

                # save the `bounce_list` to a file
                with open(path_output_folder + video_file.replace('.mp4', '_predict.txt'), 'w') as file_out:
                    for bounce in bounce_list:
                        file_out.write(str(bounce) + '\n')

                # Calculate precision, recall, and F1 score
                if num_tp + num_fp == 0:
                    precision = 0
                else:
                    precision = num_tp / (num_tp + num_fp)
                if num_tp + num_fn == 0:
                    recall = 0
                else:
                    recall = num_tp / (num_tp + num_fn)
                if precision + recall == 0:
                    f1_score = 0
                else:
                    f1_score = 2 * (precision * recall) / (precision + recall)

                # Append the evaluation results to the list
                FP_frames_str = '; '.join(map(str, FP_frames))
                FN_frames_str = '; '.join(map(str, FN_frames))
                results.append([video_file, num_tp, num_fp, num_tn, num_fn, precision, recall, f1_score, FP_frames_str, FN_frames_str])
    if len(results) == 0:
        print('Video file list {} could not be read or had no entries'.format(args.file_input_video_list))
        return

    # Create a DataFrame from the results list
    df = pd.DataFrame(results, columns=['video_filename', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1_score', 'FP_frames', 'FN_frames'])

    # Calculate the overall precision, recall, and F1 score
    if sum_tp + sum_fp == 0:
        overall_precision = 0
    else:
        overall_precision = sum_tp / (sum_tp + sum_fp)
    if sum_tp + sum_fn == 0:
        overall_recall = 0
    else:
        overall_recall = sum_tp / (sum_tp + sum_fn)
    if overall_precision + overall_recall == 0:
        overall_f1_score = 0
    else:
        overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)

    # Append the overall results to the DataFrame
    df = df.append({'video_filename': 'Overall', 'tp': sum_tp, 'fp': sum_fp, 'tn': sum_tn, 'fn': sum_fn, 'precision': overall_precision, 'recall': overall_recall, 'f1_score': overall_f1_score, 'FP_frames': 'n/a', 'FN_frames': 'n/a'},  ignore_index=True)

    # Print the overall results
    print('Overall results:')
    print('Precision: {:.4f}'.format(overall_precision))
    print('Recall: {:.4f}'.format(overall_recall))
    print('F1 Score: {:.4f}'.format(overall_f1_score))

    # Save the DataFrame to an Excel file
    df.to_excel(path_output_folder + 'results.xlsx', index=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ball_track_model', type=str, help='path to pretrained model for ball detection')
    parser.add_argument('--path_court_model', type=str, help='path to pretrained model for court detection')
    parser.add_argument('--path_bounce_model', type=str, help='path to pretrained model for bounce detection')
    parser.add_argument('--path_input_video_folder', type=str, help='path to folder containing input videos')
    parser.add_argument('--file_input_video_list', type=str, help='list file of input videos')
    parser.add_argument('--path_output_folder', type=str, help='path to output folder')
    parser.add_argument('--param_file', type=str, help='INI-style file containing parameters for the pipeline')
    parser.add_argument('--debug', action='store_true', help='debug flag')
    args = parser.parse_args()

    main(args)    





