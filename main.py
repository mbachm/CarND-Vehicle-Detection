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
from skimage.feature import hog
from scipy.ndimage.measurements import label

color_space = 'YUV'
spatial_size = (32, 32)
hist_bins = 32
orient = 11
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True
scales = [0.7, 1, 1.3, 1.5, 1.7]

def __convert_to_colorspace(img, colorspace):
    """ Function to convert colorspace to the desired one. This function
        assumes that images where read with cv2.imread. So the original
        colorspace is BGR
    """
    if colorspace != 'BGR':
        if colorspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif colorspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif colorspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif colorspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif colorspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif colorspace == 'RGB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else: feature_image = np.copy(img)
    return feature_image

def extract_for_training(imgs):
    """ Function that receives a list of image paths, load each image and
        calls single_img_features for each. Saves the features for all images
        and returns all features together. Is needed for the training of
        the SVM.
    """
    features = []
    
    for image_name in imgs:
        img = cv2.imread(image_name)
        features.append(extract_features(img, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat=True, hist_feat=True, hog_feat=True))
    return features

def extract_non_car_and_car_features_for_training():
    """ Extract the non_car and car features of the training images """
    path_to_non_car_images = './training_data/non-vehicles/*/*.png'
    path_to_car_images = './training_data/vehicles/*/*.png'
    cars = glob.glob(path_to_car_images)
    notcars = glob.glob(path_to_non_car_images)
    car_features = extract_for_training(cars)
    notcar_features = extract_for_training(notcars)
    return car_features, notcar_features

def train_svm(car_features, notcar_features):
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
    car_features, notcar_features = extract_non_car_and_car_features_for_training()
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')

    t=time.time()
    svc, X_scaler = train_svm(car_features, notcar_features)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return svc, X_scaler

def bin_spatial(img, size=(32, 32)):
    """ Function to compute binned color features """
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):
    """ Function to compute color histogram features """
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """ Function to return HOG features and visualization """
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def extract_features(image, color_space='YCrCb', spatial_size=(32, 32), hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial_feat=True, hist_feat=True,
                     hog_feat=True):
    """ Function to extract image features of single image """
    features = []
    # apply color conversion if other than 'BGR'
    feature_image = __convert_to_colorspace(image, color_space)

    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        features.append(spatial_features)
    if hist_feat is True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        features.append(hist_features)
    if hog_feat is True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return np.concatenate(features)

# Define a single function that can extract features using hog sub-sampling and make predictions
def __find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    """ Searches in the given image for vehicles with the given, trained svm and scaler. """
    draw_img = np.copy(img)

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = __convert_to_colorspace(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    predictions = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                predictions.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return predictions

def __generate_heat_map(img, bbox_list):
    """ generates a heatmap out of a list of rectangles """
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    for box in bbox_list:
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heat

def __apply_threshold(heatmap, threshold):
    """ Zero out pixels below the threshold """
    heatmap[heatmap <= threshold] = 0
    return heatmap

def __draw_labeled_bboxes(img, labels):
    """ draws a rectangle on the image for given list of labels """
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

def __draw_boxes(img, bboxes, color=(0, 255, 0), thick=6):
    """ draws list of rectangles on given image """
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def search_for_vehicles(image, ystart, ystop, clf, scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    """ Searches in the given image for vehicles with the given, trained svm and scaler. """
    hot_windows = []
    window_img = np.copy(image)
    for scale in scales:
        windows = __find_cars(window_img, ystart, ystop, scale, clf, scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        hot_windows.extend(windows)

    heatmap = __generate_heat_map(np.copy(image), hot_windows)
    heat = __apply_threshold(heatmap, 3)
    heat = np.clip(heat, 0, 255)
    labels = label(heat)
    heat_mapped_boxes = __draw_labeled_bboxes(np.copy(image), labels)
    search_boxes = __draw_boxes(np.copy(image), hot_windows, color=(0, 255, 0), thick=5)
    return search_boxes, heat, heat_mapped_boxes

def __run_svm_on_test_images(svc, X_scaler):
    """ Runs the given svm and scaler on the project specific test images """
    path_test_images = './test_images/*.jpg'
    test_images = glob.glob(path_test_images)
    ystart = 380
    ystop = 700
    for fname in test_images:
        img = cv2.imread(fname)
        found_boxes, heatmap, heat_mapped_image = search_for_vehicles(img, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
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
    global svc, X_scaler
    ystart = 380
    ystop = 700
    found_boxes, heatmap, heat_mapped_image = search_for_vehicles(image, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    return heat_mapped_image

training_pipeline()
test_images_pipeline()

output = 'processed_test_video.mp4'
clip = VideoFileClip('test_video.mp4')
output_clip = clip.fl_image(video_pipeline)
output_clip.write_videofile(output, audio=False)

"""
output_project = 'processed_project_video.mp4'
clip_project = VideoFileClip('project_video.mp4')
output_clip_project = clip_project.fl_image(video_pipeline)
output_clip_project.write_videofile(output_project, audio=False)
"""
