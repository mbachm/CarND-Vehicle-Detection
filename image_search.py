import numpy as np
import cv2
import feature_extraction

### Box sizes for sliding window search
xy_sizes = [(64, 64), (128, 128), (256, 256)]

def __draw_boxes(img, bboxes, color=(0, 255, 0), thick=6):
	""" draws list of rectangles on given image"""
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

def __slide_window(img, clf, scaler, x_start_stop=[None, None], y_start_stop=[None, None],
	xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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
	on_windows = []
	for window in windows:
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		features = feature_extraction.single_img_features(test_img)
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		prediction = clf.predict(test_features)
		if prediction == 1:
			on_windows.append(window)
	return on_windows

def search_for_vehicles(image, clf, scaler):
	hot_windows = []
	window_img = np.copy(image)
	for box_size in xy_sizes:
		windows = __slide_window(window_img, clf, scaler, x_start_stop=[None, None], y_start_stop=[380, 700],
			xy_window=box_size, xy_overlap=(0.5, 0.5))
		hot_windows2 = __search_windows(window_img, windows, clf, scaler)
		hot_windows.extend(hot_windows2)
	
	window_img = __draw_boxes(window_img, hot_windows, color=(0, 255, 0), thick=5)
	return window_img 