# Writeup Report

--

**Vehicle Detection Project**


[//]: # (Image References)
[test1]: ./output_images/test1.jpg "Test image 1"
[test2]: ./output_images/test2.jpg "Test image 2"
[test3]: ./output_images/test3.jpg "Test image 3"
[test4]: ./output_images/test4.jpg "Test image 4"
[test5]: ./output_images/test5.jpg "Test image 5"
[test6]: ./output_images/test6.jpg "Test image 6"
[svm]: ./output_images/training.png "Perfomance time"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup Report

My project consists of multiple python modules:
* `main.py`: The main class, where all other methods/modules are called from
* `feature_extraction.py`: Module for feature extraction of images. Additionally, it has a sliding window search for vehicle detection and generates a heatmap.
* `vehicle_detector.py`: Class to keep track of up to 3 previous heatmaps. Also stores the current one and the threshold for the heatmap.

Now I will go through each necessary rubric point and explain the details of my implementation.

### Histogram of Oriented Gradients (HOG)

#### 1. Extraction of HOG features from the training images.

In the `main.py` class there is the `training_pipeline` function (line 52 -64), which is responsible for training my support vector machine. It extracts the non_car and car features with the `__extract_non_car_and_car_features_for_training` function (line 28-36), which itselfs call `__extract_for_training`(line 15-26). There I start with reading all images with a glob and pass each image to the `extract_features` function from my `feature_extraction` module. The `extract_features` (line 60-77) function converts the given image to another colorspace and then compute binned color features, color histogram features and HOG features for all channels. The extracted features are all concatenated with numpy (line 77 in feature_extraction) and after that returned. In contrast to the lession, the colorconversion is from BGR instead of RGB to another colorspace, as I use cv2 for reading my images (`feature_extraction.py`, `__convert_to_colorspace`, lines 18.37).

#### 2. Settled final choice of HOG parameters.

I tried various combinations of parameters and final this combination worked best for my implementation:

* color_space: 'YCrCb'
* spatial_size: (32, 32)
* hist_bins: 32
* orient: 8
* pix_per_cell: 8
* cell_per_block: 2
* hog_channel: 'ALL'

They are defined in the lines from 8 to 16 in `feature_extraction.py`
This is different from the well performing parameters from the [udacity forum](https://discussions.udacity.com/t/good-tips-from-my-reviewer-for-this-vehicle-detection-project/232903/14). But with the presented parameter set I got to many false positives.

#### 3. Training my classifier using your selected HOG features and color features

I trained a linear SVM using the StandardScaler for standardation of my test features and sklearn train_test_split for training. (see function `__train_svm`, lines 38-50). I use 20 percent of the training set as test_size. The training time is about 85 seconds and I got a test accuracy of 99.24%

![Console log of svm training][svm]

### Sliding Window Search

#### Implementation of a sliding window search. How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemtned in the `__find_cars` function in `feature_extraction.py`. I used the code from lesson 35 and just changed `cells_per_step` to `1` to generate more overlaps. Furthermore I ajusted the `__get_hog_features` function as I define parameters my parameters for all of the `feature_extraction.py` functions.

I used the test code of `lesson 32 Sliding Window Implementation` and some exploration to decided which scales to use in my implementation. My final choices where the scales of `1`, `1.5` and `2.5` with the ystart and yend points of (380, 480), (400, 600) and (500, 700) (line 16) for my sliding window search.

The sliding window search is used in the function `search_for_vehicles` (lines 174-190), which is called from the `video_pipeline` function of the main module (line 98-102).

The sliding window search returns for all three scales potential vehicle detections as a tupel. Afterwards, I apply a heatmap with a threshold of 13 to all potential vehicles (line 182, feature_extraction). The generated heatmap is stored within my `vehicleDetector` class. This class stores the last 3 heatmaps for a better accuracy in the video_pipeline.

#### Some examples of test images to demonstrate how my pipeline is working.

| Pipeline steps for images |
|:---:|
|![Demonstration of pipeline for image1][test1]|
|![Demonstration of pipeline for image2][test2]|
|![Demonstration of pipeline for image3][test3]|
|![Demonstration of pipeline for image4][test4]|
|![Demonstration of pipeline for image5][test5]|
|![Demonstration of pipeline for image6][test6]|

As you can see, the heatmap removes removes some false positives (e.g. test image 5). I had to modify the threshold of the heatmap multiply times to get a good result.

---

### Video Implementation

#### Link to my final video output. 

Here's a [link to my video result](./processed_project_video.mp4)


#### Filter for false positives and some method for combining overlapping bounding boxes.

As stated above, I create a heatmap and then apply a threshold of each positive detection in each frame of the video to identify vehicle positions and store this heatmap for the last 3 frames. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected (lines 184-189).

---

### Discussion

#### Problems and issues I faced in my implementation of this project.

I had a lot of issues to find the right combination of extraction parameters. With the wrong combination my SVM was not able to find the white car in the video or I got to many false positives.

As you can see in the also provided [processed test video](./processed_test_video.mp4), my pipeline also detects the yellow as potential car. Perhaps better extraction parameters or another kernel for the svm could prevent my pipeline or a higher threshold could prevent my pipeline from this false detection. Another approach could be to use Keras as my classifier to generate a better result.
