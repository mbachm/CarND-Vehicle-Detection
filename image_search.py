import numpy as np
import cv2

xy_sizes = [(64, 64), (128, 128), (256, 256)]

def __draw_boxes(img, bboxes, color=(0, 255, 0), thick=6):
	imcopy = np.copy(img)
	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	return imcopy

def __slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
	xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	# Compute the span of the region to be searched    
	x_span = x_start_stop[1] - x_start_stop[0]
	y_span = y_start_stop[1] - y_start_stop[0]
	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	# Compute the number of windows in x/y
	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
	nx_windows = np.int((x_span-nx_buffer)/nx_pix_per_step) 
	ny_windows = np.int((y_span-ny_buffer)/ny_pix_per_step) 
	# Initialize a list to append window positions to
	window_list = []
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	# Return the list of windows
	return window_list

def search_for_vehicles(image):
	window_img = np.copy(image)
	for box_size in xy_sizes:
		windows = __slide_window(window_img, x_start_stop=[None, None], y_start_stop=[380, 700], 
			xy_window=box_size, xy_overlap=(0.5, 0.5))       
		window_img = __draw_boxes(window_img, windows, color=(0, 255, 0), thick=6) 
	return window_img 