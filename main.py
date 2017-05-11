import numpy as np
import cv2
import glob
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
import feature_extraction as fe
from vehicleDetector import vehicleDetector
from collections import deque

def __extract_for_training(imgs):
    """ Function that receives a list of image paths, load each image and
        calls single_img_features for each. Saves the features for all images
        and returns all features together. Is needed for the training of
        the SVM.
    """
    features = []
    
    for image_name in imgs:
        img = cv2.imread(image_name)
        features.append(fe.extract_features(img))
    return features

def __extract_non_car_and_car_features_for_training():
    """ Extract the non_car and car features of the training images """
    path_to_non_car_images = './training_data/non-vehicles/*/*.png'
    path_to_car_images = './training_data/vehicles/*/*.png'
    cars = glob.glob(path_to_car_images)
    notcars = glob.glob(path_to_non_car_images)
    car_features = __extract_for_training(cars)
    notcar_features = __extract_for_training(notcars)
    return car_features, notcar_features

def __train_svm(car_features, notcar_features):
    """ Trains the svm with the given car and non-car features """
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Feature vector length:', len(X_train[0]))
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc, X_scaler

def training_pipeline():
    """ Pipeline to train svm with time measurement """
    global svc, X_scaler
    t=time.time()
    car_features, notcar_features = __extract_non_car_and_car_features_for_training()
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')

    t=time.time()
    svc, X_scaler = __train_svm(car_features, notcar_features)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return svc, X_scaler

def __run_svm_on_test_images(svc, X_scaler):
    """ Runs the given svm and scaler on the project specific test images """
    path_test_images = './test_images/*.jpg'
    test_images = glob.glob(path_test_images)
    for fname in test_images:
        img = cv2.imread(fname)
        found_boxes, heatmap, heat_mapped_image = fe.search_for_vehicles(img, svc, X_scaler, vehicleDetector())
        ### Save origial image, found boxes, heatmap and detected vehicles
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        heat_mapped_image = cv2.cvtColor(heat_mapped_image, cv2.COLOR_BGR2RGB)
        found_boxes = cv2.cvtColor(found_boxes, cv2.COLOR_BGR2RGB)
        f, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(found_boxes)
        ax2.set_title('Found boxes', fontsize=50)
        ax3.imshow(heatmap, cmap='hot')
        ax3.set_title('Heatmap', fontsize=50)
        ax4.imshow(heat_mapped_image)
        ax4.set_title('Detected vehicles', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig('./output_images/'+fname.split('/')[-1], dpi=100)

def test_images_pipeline():
    global svc, X_scaler
    """ Pipeline to run svm against test images with time measurement """
    t=time.time()
    __run_svm_on_test_images(svc, X_scaler)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to search in all testimages for vehicles')

def video_pipeline(image):
    """ Pipeline for project video """
    global svc, X_scaler, vehicleDetector
    found_boxes, heatmap, heat_mapped_image = fe.search_for_vehicles(image, svc, X_scaler, vehicleDetector)
    return heat_mapped_image

training_pipeline()
test_images_pipeline()

vehicleDetector = vehicleDetector()
output = 'processed_test_video.mp4'
clip = VideoFileClip('test_video.mp4')
output_clip = clip.fl_image(video_pipeline)
output_clip.write_videofile(output, audio=False)

### Clear heat images cache
vehicleDetector.heat_images = deque(maxlen=3)

output_project = 'processed_project_video.mp4'
clip_project = VideoFileClip('project_video.mp4')
output_clip_project = clip_project.fl_image(video_pipeline)
output_clip_project.write_videofile(output_project, audio=False)
