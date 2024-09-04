import cv2
import numpy as np
from bounce_detector import BounceDetector
from ball_detector import BallDetector
from court_detection_net import CourtDetectorNet
import argparse
import torch
import pandas as pd
from datetime import datetime
import os

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


def run_detection(path_to_video_file, video_file, ball_detector, bounce_detector, court_detector):
    """
    Run the detection process on a video file.

    Parameters:
    - video_file (str): The path to the video file.
    - ball_detector (object): The ball detector object.
    - bounce_detector (object): The bounce detector object.
    - court_detector (object): The court detector object.
    - path_output_folder (str): The path to the output folder.
    - args (dict): Additional arguments.

    Returns:
    - bounce_list (list): A list of integers, where 1 (0) represents the presence (absence) of a bounce in each frame.
    """
    video_filepath = path_to_video_file + '/' + video_file

    # Read the video file
    frames, fps = read_video(video_filepath)

    # Run the detection process
    ball_track = ball_detector.infer_model(frames, video_file)

    # Predict bounces
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    # Generate a list corresponding to each frame, where 1 (0) represents the presence (absence) of a bounce in the frame    
    bounce_list = [1 if i in bounces else 0 for i in range(len(frames))]
    return bounce_list

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
                print("Processing video file: {}".format(video_file))
                bounce_list = run_detection(args.path_input_video_folder, video_file, 
                                            ball_detector, 
                                            bounce_detector,
                                            court_detector)
                bounce_gt_list = get_ground_truth(args.path_input_video_folder + '/' + video_file, args)
                num_tp, num_fp, num_tn, num_fn, FP_frames, FN_frames = evaluate(bounce_list, bounce_gt_list, args)
                sum_tp += num_tp
                sum_fp += num_fp
                sum_tn += num_tn
                sum_fn += num_fn

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

                # Append the results to the list
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
    parser.add_argument('--path_bounce_model', type=str, help='path to pretrained model for bounce detection')
    parser.add_argument('--path_court_model', type=str, help='path to pretrained model for court detection')
    parser.add_argument('--file_input_video_list', type=str, help='list file of input videos')
    parser.add_argument('--path_output_folder', type=str, help='path to output folder')
    parser.add_argument('--param_file', type=str, help='INI-style file containing parameters for the pipeline')
    parser.add_argument('--debug', action='store_true', help='debug flag')
    args = parser.parse_args()

    main(args)
    






