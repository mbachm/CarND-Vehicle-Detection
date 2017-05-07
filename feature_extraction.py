import numpy as np
import cv2
from skimage.feature import hog

def __convert_to_colorspace(img, cspace='RGB'):
	""" Function to convert colorspace to the desired one. This function
		assumes that images where read with cv2.imread. So the original
		colorspace is BGR
	"""
	if cspace != 'BGR':
		if cspace == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		elif cspace == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
		elif cspace == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
		elif cspace == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		elif cspace == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
		elif cpace == 'RGB':
			feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		else: feature_image = np.copy(img)
	return feature_image

# Define a function to compute binned color features  
def __bin_spatial(img, size=(32, 32)):
	color1 = cv2.resize(img[:,:,0], size).ravel()
	color2 = cv2.resize(img[:,:,1], size).ravel()
	color3 = cv2.resize(img[:,:,2], size).ravel()
	return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features  
def __color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

# Define a function to return HOG features and visualization
def __get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	# Call with two outputs if vis==True
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

def single_img_features(img, cspace='RGB', spatial_size=(32, 32), 
	hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8,
	cell_per_block=2, hog_channel=0):
	feature_image = __convert_to_colorspace(img, cspace)
	spatial_features = __bin_spatial(feature_image, size=spatial_size)
	hist_features = __color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
	if hog_channel == 'ALL':
		hog_features = []
		for channel in range(feature_image.shape[2]):
			hog_features.append(__get_hog_features(feature_image[:,:,channel],
				orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False))
		hog_features = np.ravel(hog_features)        
	else:
		hog_features = np.ravel(__get_hog_features(feature_image[:,:,hog_channel], orient,
			pix_per_cell, cell_per_block, vis=False, feature_vec=False))
	return np.concatenate((spatial_features, hist_features, hog_features))



def extract_for_training(imgs, cspace='RGB', spatial_size=(32, 32),
	hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8,
	cell_per_block=2, hog_channel=0):
	# Create a list to append feature vectors to
	features = []
	
	for image_name in imgs:
		img = cv2.imread(image_name)
		features.append(single_img_features(img, cspace, spatial_size, hist_bins, hist_range,
			orient, pix_per_cell, cell_per_block, hog_channel))
	return features
