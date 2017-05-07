import numpy as np
import cv2
from scipy.ndimage.measurements import label
import feature_extraction
import heatmap_history

### Box sizes for sliding window search
xy_sizes = [(32,32), (64, 64), (96, 96), (128, 128), (192, 192), (256, 256)]

def __draw_boxes(img, bboxes, color=(0, 255, 0), thick=6):
	""" draws list of rectangles on given image """
	imcopy = np.copy(img)
	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	return imcopy

def __get_start_stop_and_nperstep_and_nwindows(start_stop=[None, None], shape=720,
	window=64, overlap=0.5):
	""" Calculates the start_stop array and number of pixel per step and
		number of windows for the sliding window search on behalf of the given
		parameters.
	"""
	if start_stop[0] == None:
		start_stop[0] = 0
	if start_stop[1] == None:
		start_stop[1] = shape
	span = start_stop[1] - start_stop[0]
	n_pix_per_step = np.int(window*(1 - overlap))
	n_buffer = np.int(window*(overlap))
	n_windows = np.int((span-n_buffer)/n_pix_per_step)
	return start_stop, n_pix_per_step, n_windows

def __slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
	xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	""" Create list of boxes with given parameters for a sliding window search
		for the given image. Startpoints (x, y) for boxcreations, overlapping
		parameters and size of the boxes have to be defined.
	"""
	x_start_stop, nx_pix_per_step, nx_windows = __get_start_stop_and_nperstep_and_nwindows(x_start_stop,
		img.shape[1], xy_window[0], xy_overlap[0])
	y_start_stop, ny_pix_per_step, ny_windows = __get_start_stop_and_nperstep_and_nwindows(y_start_stop,
		img.shape[0], xy_window[1], xy_overlap[1])

	window_list = []
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			window_list.append(((startx, starty), (endx, endy)))
	return window_list


def __search_windows(img, windows, clf, scaler):
	""" Checks if the trained svm and scaler detect a vehicle in the boxes of
		of the given image.
	"""
	on_windows = []
	for window in windows:
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		features = feature_extraction.single_img_features(test_img)
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		prediction = clf.predict(test_features)
		if prediction == 1:
			on_windows.append(window)
	return on_windows

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

def __get_boxes_of_labels(labels):
	boxes = []
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		boxes.append(bbox)
	return boxes

def __draw_labeled_bboxes(img, boxes):
	""" draws a rectangle on the image for given list of labels """
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		boxes.append(bbox)
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	return boxes, img

def search_for_vehicles(image, clf, scaler, heatmap_history):
	""" Searches in the given image for vehicles with the given, trained svm
		and scaler.
	"""
	### TODO: Add Heatmap_history
	hot_windows = []
	window_img = np.copy(image)
	for box_size in xy_sizes:
		windows = __slide_window(window_img, x_start_stop=[None, None], y_start_stop=[380, 700],
			xy_window=box_size, xy_overlap=(0.5, 0.5))
		hot_windows.extend(__search_windows(window_img, windows, clf, scaler))
	heatmap = __generate_heat_map(np.copy(image), hot_windows)
	heat = __apply_threshold(heatmap, 1)
	heat = np.clip(heat, 0, 255)
	labels = label(heat)
	boxes = __get_boxes_of_labels(labels)
	heatmap_history.add_data(boxes)
	heat_mapped_boxes = __draw_boxes(np.copy(image), heatmap_history.best_fit_boxes, color=(0, 0, 255), thick=5)
	search_boxes = __draw_boxes(np.copy(image), hot_windows, color=(0, 255, 0), thick=5)
	return search_boxes, heat, heat_mapped_boxes
