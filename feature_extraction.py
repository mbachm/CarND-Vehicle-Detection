import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import vehicle_detector
from collections import deque

### Parameters
color_space = 'YCrCb'
spatial_size = (32, 32)
hist_bins = 32
orient = 8
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
scales = [(380, 480, 1), (400, 600, 1.5), (500, 700, 2.5)]

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

def __bin_spatial(img):
    """ Function to compute binned color features """
    color1 = cv2.resize(img[:, :, 0], spatial_size).ravel()
    color2 = cv2.resize(img[:, :, 1], spatial_size).ravel()
    color3 = cv2.resize(img[:, :, 2], spatial_size).ravel()
    return np.hstack((color1, color2, color3))

def __color_hist(img):
    """ Function to compute color histogram features """
    channel1_hist = np.histogram(img[:,:,0], bins=hist_bins)
    channel2_hist = np.histogram(img[:,:,1], bins=hist_bins)
    channel3_hist = np.histogram(img[:,:,2], bins=hist_bins)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def __get_hog_features(img, feature_vec=True):
    """ Function to return HOG features without visualization """
    return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
        visualise=False, feature_vector=feature_vec)

def extract_features(image):
    """ Function to extract image features of single image """
    features = []
    # apply color conversion if other than 'BGR'
    feature_image = __convert_to_colorspace(image, color_space)

    features.append(__bin_spatial(feature_image))
    features.append(__color_hist(feature_image))

    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(__get_hog_features(feature_image[:, :, channel], feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = __get_hog_features(feature_image[:, :, hog_channel], feature_vec=True)
    features.append(hog_features)
    return np.concatenate(features)

# Define a single function that can extract features using hog sub-sampling and make predictions
def __find_cars(img, ystart, ystop, scale, svc, X_scaler):
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
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = __get_hog_features(ch1, feature_vec=False)
    hog2 = __get_hog_features(ch2, feature_vec=False)
    hog3 = __get_hog_features(ch3, feature_vec=False)
    
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
            spatial_features = __bin_spatial(subimg)
            hist_features = __color_hist(subimg)
            
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

def search_for_vehicles(image, clf, scaler, vehicleDetector):
    """ Searches in the given image for vehicles with the given, trained svm and scaler. """
    hot_windows = []
    window_img = np.copy(image)
    for ystart, ystop, scale in scales:
        windows = __find_cars(window_img, ystart, ystop, scale, clf, scaler)
        hot_windows.extend(windows)

    vehicleDetector.add_heatmap_and_threshold(window_img, hot_windows)

    heatmap = __generate_heat_map(np.copy(image), hot_windows)
    heat = __apply_threshold(heatmap, vehicleDetector.threshold_to_apply)
    heat = np.clip(heat, 0, 255)
    labels = label(vehicleDetector.heatmap)
    heat_mapped_boxes = __draw_labeled_bboxes(np.copy(image), labels)
    search_boxes = __draw_boxes(np.copy(image), hot_windows, color=(0, 255, 0), thick=5)
    return search_boxes, heat, heat_mapped_boxes
