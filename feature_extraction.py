import numpy as np
import cv2
from skimage.feature import hog

def __read_image_and_convertion_to_color(image, cspace='RGB'):
	img = cv2.imread(image)
	feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if cspace != 'RGB':
		if cspace == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif cspace == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif cspace == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif cspace == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif cspace == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(img)
	return feature_image

# Define a function to compute binned color features  
def __bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features

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

def extract(imgs, cspace='RGB', spatial_size=(32, 32),
	hist_bins=32, hist_range=(0, 256), orient=9, 
	pix_per_cell=8, cell_per_block=2, hog_channel=0):
	# Create a list to append feature vectors to
	features = []
	
	for image_name in imgs:
		feature_image = __read_image_and_convertion_to_color(image_name, cspace)
		
		spatial_features = __bin_spatial(feature_image, size=spatial_size)
		hist_features = __color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
		
		# Call get_hog_features() with vis=False, feature_vec=True
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(__get_hog_features(feature_image[:,:,channel], 
					orient, pix_per_cell, cell_per_block, 
					vis=False, feature_vec=True))
			hog_features = np.ravel(hog_features)        
		else:
			hog_features = __get_hog_features(feature_image[:,:,hog_channel], orient, 
				pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		# Append the new feature vector to the features list
		features.append(np.concatenate((hog_features, spatial_features, hist_features)))
	return features
