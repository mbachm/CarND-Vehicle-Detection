import numpy as np
import cv2
from skimage.feature import hog

# Standard parameters for feature extraction
colorspace = 'YUV'
spatial_size = (8, 8)
hist_bin = 32
hist_range = (0, 256)
orientations = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'


def set_parameters(cspace, s_size, h_bins, h_range, orient, pix_cell,
	cell_block, h_channel):
	global colorspace
	colorspace = cspace
	global spatial_size
	spatial_size = s_size
	global histbin
	hist_bin = h_bins
	global hist_range
	hist_range = h_range
	global orientations
	orientations = orient
	global pix_per_cell
	pix_per_cell = pix_cell
	global cell_per_block
	cell_per_block = cell_block
	global hog_channel
	hog_channel = h_channel

def __convert_to_colorspace(img):
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

# Define a function to compute binned color features  
def __bin_spatial(img):
	color1 = cv2.resize(img[:,:,0], spatial_size).ravel()
	color2 = cv2.resize(img[:,:,1], spatial_size).ravel()
	color3 = cv2.resize(img[:,:,2], spatial_size).ravel()
	return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features  
def __color_hist(img):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=hist_bin, range=hist_range)
	channel2_hist = np.histogram(img[:,:,1], bins=hist_bin, range=hist_range)
	channel3_hist = np.histogram(img[:,:,2], bins=hist_bin, range=hist_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

# Define a function to return HOG features and visualization
def __get_hog_features(img, vis=False):
	if vis == True:
		features, hog_image = hog(img, orientations=orientations, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
			visualise=vis, feature_vector=False)
		return features, hog_image
	else:      
		features = hog(img, orientations=orientations, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
			visualise=vis, feature_vector=False)
		return features

def single_img_features(img):
	feature_image = __convert_to_colorspace(img)
	spatial_features = __bin_spatial(feature_image)
	hist_features = __color_hist(feature_image)
	if hog_channel == 'ALL':
		hog_features = []
		for channel in range(feature_image.shape[2]):
			hog_features.append(__get_hog_features(feature_image[:,:,channel],
				vis=False))
		hog_features = np.ravel(hog_features)        
	else:
		hog_features = np.ravel(__get_hog_features(feature_image[:,:,hog_channel],
			vis=False))
	return np.concatenate((spatial_features, hist_features, hog_features))



def extract_for_training(imgs):
	# Create a list to append feature vectors to
	features = []
	
	for image_name in imgs:
		img = cv2.imread(image_name)
		features.append(single_img_features(img))
	return features
