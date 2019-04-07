import cv2
import numpy as np

from skimage.feature import hog
from scipy.ndimage.measurements import label

class VehicleDetector:

    class DetectionHistory:
        # Define a class to store data from video
        ## This logic is inspired from Harish Vadlamani
        def __init__(self):
            self.prev_rects = []

        def add_rects(self, rects):
            self.prev_rects.append(rects)
            if len(self.prev_rects) > 13:
                 self.prev_rects = self.prev_rects[len(self.prev_rects)-13:]

        def clear(self):
            self.prev_rects = []

    def __convert_color(self, img):
        # apply color conversion if other than 'RGB'
        if self.__color_space != 'RGB':
            if self.__color_space == 'HSV':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.__color_space == 'LUV':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.__color_space == 'HLS':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.__color_space == 'YUV':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.__color_space == 'YCrCb':
                converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            converted_image = np.copy(img)

        return converted_image

    def __get_hog_features(self, img, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(img, orientations=self.__orient,
                                      pixels_per_cell=(self.__pix_per_cell, self.__pix_per_cell),
                                      cells_per_block=(self.__cell_per_block, self.__cell_per_block),
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.__orient,
                           pixels_per_cell=(self.__pix_per_cell, self.__pix_per_cell),
                           cells_per_block=(self.__cell_per_block, self.__cell_per_block),
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    def __bin_spatial(self, img):
        return cv2.resize(img, self.__spatial_size).ravel()

    def __color_hist(self, img):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.__hist_bins, range=(0, 256))
        channel2_hist = np.histogram(img[:, :, 1], bins=self.__hist_bins, range=(0, 256))
        channel3_hist = np.histogram(img[:, :, 2], bins=self.__hist_bins, range=(0, 256))

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

        # Return the individual histograms, bin_centers and feature vector
        return hist_features.ravel()

    def __find_cars(self, img, ystart, ystop, xstart, xstop, scale, cells_per_step):
        detection_boxes = []

        draw_img = np.copy(img)
        img_tosearch = img[ystart:ystop, xstart:xstop, :]

        ctrans_tosearch = self.__convert_color(img_tosearch)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        if self.__hog_channel == 'ALL':
            ch1 = ctrans_tosearch[:, :, 0]
            ch2 = ctrans_tosearch[:, :, 1]
            ch3 = ctrans_tosearch[:, :, 2]
        else:
            ch1 = ctrans_tosearch[:, :, self.__hog_channel]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.__pix_per_cell) - self.__cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.__pix_per_cell) - self.__cell_per_block + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.__pix_per_cell) - self.__cell_per_block + 1
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = self.__get_hog_features(ch1, feature_vec=False)
        if self.__hog_channel == 'ALL':
            hog2 = self.__get_hog_features(ch2, feature_vec=False)
            hog3 = self.__get_hog_features(ch3, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                # Extract HOG for this patch
                if self.__hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

                xleft = xpos*self.__pix_per_cell
                ytop = ypos*self.__pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))

                # Get color features
                spatial_features = self.__bin_spatial(subimg)
                hist_features = self.__color_hist(subimg)

                # Scale features and make a prediction
                X = np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1).astype(np.float64)
                test_features = self.__X_scaler.transform(X)
                test_prediction = self.__clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)+xstart
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart), (255, 0, 0), 6)
                    detection_boxes.append(((int(xbox_left), int(ytop_draw+ystart)), (int(xbox_left+win_draw), int(ytop_draw+win_draw+ystart))))

        return detection_boxes

    def __add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def __apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def __draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

        # Return the image
        return img

    def pipeline(self, img):
        # convert video bgr image to rgb
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # execure lane detection and get undistorted image
        lane_image, undist_image = self.__lane_detector.pipeline(image)
        search_data = [
            {"ystart": 400, "ystop": 650, "xstart": 400, "xstop": 1280, "scale": 2.0, "cells_per_step": 2},
            {"ystart": 400, "ystop": 500, "xstart": 400, "xstop": 1280, "scale": 1.5, "cells_per_step": 2},
            {"ystart": 400, "ystop": 650, "xstart": 400, "xstop": 1280, "scale": 2.0, "cells_per_step": 2},
            {"ystart": 400, "ystop": 500, "xstart": 400, "xstop": 1280, "scale": 1.5, "cells_per_step": 2},
            {"ystart": 400, "ystop": 460, "xstart": 400, "xstop": 1280, "scale": 0.75, "cells_per_step": 2}
        ]

        # search for vehicles in image
        boxes = []
        for data in search_data:
            boxes.append(self.__find_cars(np.copy(undist_image), data["ystart"], data["ystop"], data["xstart"], data["xstop"], data["scale"], data["cells_per_step"]))

        boxes = [item for sublist in boxes for item in sublist]

        if len(boxes) > 0:
            self.__detection_history.add_rects(boxes)

            heatmap_img = np.zeros_like(img[:, :, 0])
            for rect_set in self.__detection_history.prev_rects:
                heatmap_img = self.__add_heat(heatmap_img, rect_set)

            heatmap_img = self.__apply_threshold(heatmap_img, 1 + len(self.__detection_history.prev_rects)//2)
            labels = label(heatmap_img)
            heat_image = self.__draw_labeled_bboxes(cv2.cvtColor(lane_image, cv2.COLOR_RGB2BGR), labels)

            return heat_image

        else:
            self.__detection_history.clear()
            return cv2.cvtColor(lane_image, cv2.COLOR_RGB2BGR)

    def __init__(self, lane_detector, clf, X_scaler, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(16, 16), hist_bins=32, hog_channel="ALL", color_space="YUV"):
        self.__detection_history = VehicleDetector.DetectionHistory()
        self.__lane_detector = lane_detector
        self.__clf = clf
        self.__X_scaler = X_scaler
        self.__orient = orient
        self.__pix_per_cell = pix_per_cell
        self.__cell_per_block = cell_per_block
        self.__spatial_size = spatial_size
        self.__hist_bins = hist_bins
        self.__hog_channel = hog_channel
        self.__color_space = color_space
    pass
