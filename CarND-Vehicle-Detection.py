import sys
import glob
import pickle
import os.path
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

from LaneDetector import create_lane_detector
from VehicleDetector import VehicleDetector
from features import extract_features
from search import slide_window
from draw import draw_labeled_bboxes
from search import find_cars
from search import add_heat
from search import apply_threshold

def get_training_data(image_path_pattern, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block,
                      hog_channel, spatial_feat, hist_feat, hog_feat):
    # get images from path
    images = glob.glob(image_path_pattern, recursive=True)

    # initialize arrays
    cars = []
    not_cars = []

    # append image path to array
    for image in images:
        if 'non-vehicles' in image:
            not_cars.append(image)
        else:
            cars.append(image)

    # Temporär um die Daten zu begrenzen
    #sample_size = 500
    #cars = cars[0:sample_size]
    #not_cars = not_cars[0:sample_size]

    print('Car Images: ' + str(len(cars)))
    print('Non-Car Images: ' + str(len(not_cars)))

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(not_cars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    return X, y

def main(argv):

    ML_MODEL = 'model.pickle'
    image_path_pattern = 'train_images/**/*.png'
    color_space = 'YUV'
    spatial_size = (64, 64)
    hist_bins = 32
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = "ALL"
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    # load existing SVM classifier and X scaler
    if os.path.isfile(ML_MODEL):
        # if model exists, load classifier
        with open(ML_MODEL, 'rb') as handle:
            model = pickle.load(handle)
            clf = model["clf"]
            X_scaler = model["xscaler"]
            color_space = model["color_space"]
            spatial_size = model["spatial_size"]
            hist_bins = model["hist_bins"]
            orient = model["orient"]
            pix_per_cell = model["pix_per_cell"]
            cell_per_block = model["cell_per_block"]
            hog_channel = model["hog_channel"]
            spatial_feat = model["spatial_feat"]
            hist_feat = model["hist_feat"]
            hog_feat = model["hog_feat"]

    else:
        # otherwise create a new classifier
        # create features and labels
        t0 = time()
        X, y = get_training_data(image_path_pattern, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                                 cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
        print("load training data time: " + str(round(time()-t0, 3)) + "s")

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)

        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # split training and test data
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

        # Use a linear SVC
        t0 = time()
        clf = LinearSVC(loss='hinge')
        clf.fit(X_train, y_train)
        print("training time: " + str(round(time()-t0, 3)) + "s")

        # execute test prediction
        t0 = time()
        pred = clf.predict(X_test)
        print("predict time: " + str(round(time()-t0, 3)) + "s")

        # print accuracy
        accuracy = clf.score(X_test, y_test)
        print("Accuracy: " + str(accuracy) + "s")

        # store classifier and scaler
        model = {"clf": clf,
                 "xscaler": X_scaler,
                 "color_space": color_space,
                 "spatial_size": spatial_size,
                 "hist_bins": hist_bins,
                 "orient": orient,
                 "pix_per_cell": pix_per_cell,
                 "cell_per_block": cell_per_block,
                 "hog_channel": hog_channel,
                 "spatial_feat": spatial_feat,
                 "hist_feat": hist_feat,
                 "hog_feat": hog_feat}

        with open(ML_MODEL, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # create sliding windows search sample image
    y_start_stop = [None, None]
    image = mpimg.imread('test_images/test1.jpg')
    draw_image = np.copy(image)
    rects, window_img = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5), vis=True)
    cv2.imwrite("output_images/sliding_window.jpg", cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR))

    # find cars sample image
    draw_image = mpimg.imread('test_images/test1.jpg')
    boxes = []
    boxes, draw_image = find_cars(draw_image, 400, 660, 2.0, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space, vis=True)
    boxes, draw_image = find_cars(draw_image, 400, 500, 1.5, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space, vis=True)
    boxes, draw_image = find_cars(draw_image, 400, 650, 2.0, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space, vis=True)
    boxes, draw_image = find_cars(draw_image, 400, 500, 1.5, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space, vis=True)
    boxes, draw_image = find_cars(draw_image, 400, 460, 0.75, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space, vis=True)
    cv2.imwrite("output_images/find_cars.jpg", cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR))

    # Pipeline Test images
    for image_p in glob.glob('test_images/test*.jpg'):
        filename = image_p.replace("test_images\\", "").replace(".jpg", "")

        image = cv2.imread(image_p)
        draw_image = np.copy(image)

        boxes = []
        boxes += find_cars(draw_image, 400, 660, 2.0, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space)
        boxes += find_cars(draw_image, 400, 500, 1.5, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space)
        boxes += find_cars(draw_image, 400, 650, 2.0, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space)
        boxes += find_cars(draw_image, 400, 500, 1.5, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space)
        boxes += find_cars(draw_image, 400, 460, 0.75, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space)

        box_image = np.copy(image)
        for box in boxes:
            cv2.rectangle(box_image, box[0], box[1], (0, 0, 255), 6)

        cv2.imwrite('output_images/' + filename + ".jpg", box_image)

        # Add heat to each box in box list
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = add_heat(heat, boxes)
        plt.imsave('output_images/' + filename + "_heat.jpg", heat, cmap='hot')

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 3)
        plt.imsave('output_images/' + filename + "_thres.jpg", heat, cmap='hot')

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        plt.imsave('output_images/' + filename + "_labeled.jpg", labels[0], cmap='gray')
        heat_image = draw_labeled_bboxes(np.copy(image), labels)

    # Video Pipeline
    lane_detector = create_lane_detector()
    vehicle_detector = VehicleDetector(lane_detector, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, color_space)

    # write video
    output = 'output_videos/test_video.mp4'
    clip1 = VideoFileClip("test_video.mp4")
    white_clip = clip1.fl_image(vehicle_detector.pipeline)
    white_clip.write_videofile(output, audio=False)

    #output = 'output_videos/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(vehicle_detector.pipeline)
    white_clip.write_videofile(output, audio=False)

    pass

if __name__ == "__main__":
    main(sys.argv)