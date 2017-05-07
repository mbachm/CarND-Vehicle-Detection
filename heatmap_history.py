import numpy as np
from collections import deque

### pixel margin for boxes
margin = 100

### for detected heatmaps
class Heatmap_history:
	def __init__(self, queue_length=7):
		# length of queue to store data
		self.queue_length = queue_length
		#
		self.current_boxes = [np.array([False])]
		# values of recent fit
		self.recent_boxes = deque([],maxlen=queue_length)
		#
		self.best_fit_boxes = None

	def __check_box_is_within_margin(self, current_box, recent_box):
		"""
		print(recent_box[0][0] - margin)
		print(recent_box[0][1] - margin)
		print(recent_box[1][0] + margin)
		print(recent_box[1][1] + margin)
		print()
		print(current_box[0][0])
		print(current_box[0][1])
		print(current_box[1][0])
		print(current_box[1][1])
		print()
		print()
		"""
		if (((recent_box[0][0] - margin) < current_box[0][0]) and
			((recent_box[0][1] - margin) < current_box[0][1]) and
			((recent_box[1][0] + margin) < current_box[1][0]) and
			((recent_box[1][1] + margin) < current_box[1][1])):
			return True
		return False

	def __return_bigger_box(self, box1, box2):
		x_dif = (box1[0][1] - box1[0][0]) - (box2[0][1] - box2[0][0])
		y_dif = (box1[1][1] - box1[1][0]) - (box2[1][1] - box2[1][0])
		if (x_dif + y_dif) >= 0:
			return box1
		else:
			return box2

	def __set_best_fit_boxes(self):
		if self.best_fit_boxes:
			good_heat_maps = []
			for current_box in self.current_boxes:
				count = 0
				biggest_box = current_box
				for box_list in self.recent_boxes:
					for box in box_list:
						if self.__check_box_is_within_margin(current_box, box):
							biggest_box = self.__return_bigger_box(biggest_box, self.__return_bigger_box(current_box,box))
							count += 1
				### Box seems to be appear more than once
				if count > 2:
					good_heat_maps.append(biggest_box)
			self.best_fit_boxes = good_heat_maps
		else:
			## initial run. initalize boxes
			self.best_fit_boxes = self.current_boxes

	def add_data(self, heat_mapped_boxes):
		self.current_boxes = heat_mapped_boxes
		self.recent_boxes.appendleft(heat_mapped_boxes)
		self.__set_best_fit_boxes()
