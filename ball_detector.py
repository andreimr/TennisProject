from tracknet import BallTrackerNet
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

class BallDetector:
    def __init__(self, path_model=None, 
                 device='cuda', 
                 max_dist=80, 
                 binary_threshold=127, 
                 hough_min_dist=1, 
                 hough_param1=50, 
                 hough_param2=2, 
                 hough_min_radius=2, 
                 hough_max_radius=7):
        """
        Initializes the BallDetector object.
        Args:
            path_model (str, optional): Path to the pre-trained model. Defaults to None.
            device (str, optional): Device to use for model inference. Defaults to 'cuda'.
            max_dist (int, optional): Maximum distance for ball detection. Defaults to 80.
            binary_threshold (int, optional): Threshold value for binary image conversion. Defaults to 127.
            hough_min_dist (int, optional): Minimum distance between detected circles. Defaults to 1.
            hough_param1 (int, optional): the higher threshold of the two passed to the Canny edge detector 
                                            (the lower one is twice smaller). Defaults to 50.
            hough_param2 (int, optional): S the accumulator threshold for the circle centers at the detection stage. 
                                            The smaller it is, the more false circles may be detected. Circles, 
                                            corresponding to the larger accumulator values, will be returned first. 
                                            Defaults to 2.
            hough_min_radius (int, optional): Minimum radius of the detected circles. Defaults to 2.
            hough_max_radius (int, optional): Maximum radius of the detected circles. Defaults to 7.
        """
        
        self.device = device
        self.max_dist = max_dist
        self.binary_threshold = binary_threshold
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.hough_min_radius = hough_min_radius
        self.hough_max_radius = hough_max_radius    

        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()

        # downscaled resolution for the model
        self.width = 640
        self.height = 360
        # scaling factor for the detected ball points
        self.scale = 2

    def infer_model(self, frames):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
        """
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        for num in tqdm(range(2, len(frames))):
            # a.r. optimization issue: it's inefficient to resize the same image up to three times.
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num-1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num-2], (self.width, self.height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            feature_map = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(feature_map, prev_pred)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
        return ball_track

    def postprocess(self, feature_map, prev_pred):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
        :return
            x,y ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)

        # 127 is a threshold to binarize the heatmap.
        ret, heatmap = cv2.threshold(feature_map, self.binary_threshold, 255, cv2.THRESH_BINARY)

        # The HoughCircles function takes several parameters to control the circle detection process. 
        #
        # heatmap: This is the input image on which the circle detection is performed. It is likely that heatmap is a grayscale image or a single channel image where circles are expected to be present.
        # cv2.HOUGH_GRADIENT: This parameter specifies the method used for circle detection. In this case, the Hough gradient method is being used. This method is based on the gradient information of the image.
        # dp=1: This is the inverse ratio of the accumulator resolution to the image resolution. It determines the resolution of the accumulator used for circle detection. A smaller value of dp will result in a higher resolution and more accurate circle detection, but it will also increase the computational cost.
        # minDist=1: This parameter specifies the minimum distance between the centers of detected circles. If multiple circles are found within this distance, only the strongest one will be returned. Adjusting this parameter can help filter out overlapping circles.
        # param1=50: This is a threshold parameter for the edge detection stage of the circle detection process. It determines the sensitivity of the edge detection. Higher values of param1 will result in fewer circles being detected, while lower values will result in more circles being detected.
        # param2=2: This is another threshold parameter, but it is used for the center detection stage of the circle detection process. It determines the minimum number of votes required for a circle to be detected. Higher values of param2 will result in fewer circles being detected, while lower values will result in more circles being detected.
        # minRadius=2: This parameter specifies the minimum radius of the circles to be detected. Circles with a radius smaller than this value will not be considered.
        # maxRadius=7: This parameter specifies the maximum radius of the circles to be detected. Circles with a radius larger than this value will not be considered.        
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, 
                                   minDist=self.hough_min_dist, 
                                   param1=self.hough_param1, 
                                   param2=self.hough_param2, 
                                   minRadius=self.hough_min_radius,
                                   maxRadius=self.hough_max_radius)
        
        # Choose a circle having its centre within distance `max_dist` from the previous prediction.
        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*self.scale
                    y_temp = circles[0][i][1]*self.scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < self.max_dist:
                        x, y = x_temp, y_temp
                        break                
            else:
                x = circles[0][0][0]*self.scale
                y = circles[0][0][1]*self.scale
        return x, y
