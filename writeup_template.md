**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples2/hog_visual.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/examples_det1.png
[image5]: ./output_images/heatmap.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

###Writeup / README

The entire code are mainly located in `p5.ipynb`.

**1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier**

1.1 Load data for vehicle and non-vehicle

I started by reading in all of the `vehicle` and `non-vehicle` images (cell 2). The data comes from GTI and KITTI extraction images. 8,792 images for vehicles and 8,968 images for non-vehicles. The resolution for all the images is 64x64 pixel. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

1.2 HoG feature extraction

The part of hog features extraction is lcoated in the function get_hog_features() in cell 3 of `p5.ipynb`. The following two figures are examples (Vehicle and Non-vehicle) using the blue channel of `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

**2. Parameters for HoG feature and comparison**
The main critieria for parameter selection is the trade-off between classification accurary and running time (e.g. training time in HoG+SVM classifier). The combination of parameters depends on orient, pix_per_cell, cell_per_block, hog_channel, colorspace.  The following table is the comparison on accurary and running time using different combinations. The accurary and running time are test on the HoG+SVM classfication task, which is detailed in the following sections. 

| No. | orient, pix_per_cell, cell_per_block, hog_channel, colorspace  | Classifier | Accuracy | Training Time |
| :-: | :------------------------------------------------------------: | :--------: | -------: | ------------: |
| 1   |   9,  8, 2, ALL,  YUV                                          | Linear SVC | 98.14    | 4.77          |
| 2   |   9,  8, 2, ALL,  RGB                                          | Linear SVC | 97.07    | 13.0          |
| 3   |   9,  8, 2, ALL,  HSV                                          | Linear SVC | 98.06    | 6.39          |
| 4   |   9,  8, 2, ALL,  HLS                                          | Linear SVC | 98.23    | 6.19          |
| 5   |   7,  4, 2, ALL,  YUV                                          | Linear SVC | 98.06    | 50.48         |
| 6   |   7,  4, 2, ALL,  RGB                                          | Linear SVC | 96.45    | 76.15         |
| 7   |   7,  4, 2, ALL,  HSV                                          | Linear SVC | 98.03    | 61.06         |
| 8   |   7,  4, 2, ALL,  HLS                                          | Linear SVC | 97.92    | 55.71         |
| 9   |  11, 16, 2, ALL,  YUV                                          | Linear SVC | 97.89    | 1.06          |
| 10  |  11, 16, 2, ALL,  RGB                                          | Linear SVC | 96.45    | 1.99          |
| 11  |  11, 16, 2, ALL,  HSV                                          | Linear SVC | 97.66    | 1.14          |
| 12  |  11, 16, 2, ALL,  HLS                                          | Linear SVC | 97.46    | 1.22          |

From the table above, no.9 combination (orient=11, pix_per_cell=16, cell_per_block=2, hog_channel=ALL, colorspace=YUV) is the best one in trade-off between accurary and running time among all the 12 parameters configurations. Thus, I choose this parameters combination for the following classifier training. (HOG features with parameter setting : orient=11, pix_per_cell=16, cell_per_block=2, hog_channel=ALL, colorspace=YUV). I didn't need to use any color features (color histogram or binned color features). Through HoG features, I could get satisfactory classification accuracy up to ~97.8.

The corresponding part in parameters configuration is listed in `cell 6`.

**3. Training a HoG+SVM classifier**

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First, I feed the data into function extract_features() to compute the corresponding HoG features. The HoG feature extraction uses this parameter combination (orient=11, pix_per_cell=16, cell_per_block=2, hog_channel=ALL, colorspace=YUV) to obtain HoG features in dimention 1,188. The extraction is listed in `(cell 6)` and the corresponding feature extraction function is listed function extract_features() in `cell 5`. Then a feature normalizer is applied to scale the features to zero mean and unit variance using function StandardScaler() `cell 9` to obtain scaled scaled_X. The label vector y is listed in `cell 7`.
Finally for the SVM classifier training, I randomly split the vehicle and non-vehicle data into 80% for training set (14,208 images) and 20% for test set (3,552 images) using function train_test_split() `(cell 6)`. And the SVM classifier is trained using function LinearSVC(). `(cell 11)` and obtain the test accuracy as 97.97%, which meets the demands of subsequent detection task. 

**4. Sliding Window Search**

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?



I adapted the function find_cars() `cell 12` from the lesson materials.
In this function, the searching area can be determined to ignore useless search area such as top part of the image `(line 11 in cell 12)`. Color spaces such as RGB, YUV, HLS, HSV can be specified`(line 14-21 in cell 12)`. The scale of the image will be rescaled when input parameter scale is other than 1 `(line 23-26 in cell 12)`. The searching blocks and steps are listed in `(line 36-45 in cell 12)`.  Instead of explicity determine the percentage of overlap, I define how many cells to step (e.g. 2)  `(line 43 in cell 12)`
 To reduce the time in computation HoG features for each window separately, the whole image is firstly extracted into HoG features `(in line 47-51 of cell 12)`. Then this full-image feature is divided into patches to get subsampled ones according to the size of the window (64 pixel) and loops through all the possible windows(patches) `(in line 53-67 of cell 12)`. The subsampled features are then fed to the classifier for predicting whether is belongs to vechicle or not `(in line 70 of cell 12)`. If vehicle was found, the current rectangle of the patch will be added into list rectangles as the return of the function find_cars() `(in line 72-78 of cell 12)`, which is equivalent to area overlapping.
  The method I used combines HOG feature extraction with a sliding window search, but rather than performing feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("vehicle") prediction. Then, draw the detected rectangle in image.`(cell 13)`

The following figure are the demonstration of my detection pipeline.
As the one of test images (test3.jpg) contains a vehicle in small size. I append more small scales such as scale=0.8, 0.9, 1.0, 1.1. Also, when the vehicle appears near bottom part of the image, in which the vechicle exhibits a larger size. Thus bigger scale= 3.0,3.5 are appended.
The total combinations of scale and searching area are listed in function process_frame() in `cell `. Ultimately I searched on eight scales using YUV 3-channel HOG featuree as the input feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

However, the problems of overlapping detections and obvious false positives exist as in the figure above.


To reduce the quantity of the false positives and combine overlapping detections, I choose the following approach as suggested in class materials:
1. add heatmap to discriminate the true positves and false postives and filter out latter. The correspondong part is located in
function add_heat() in `cell `, which adds "heat" (+=1) for all pixels within windows where a positive detection is reported by my classifier. The individual heat-maps for the above images look like the following figure:

![alt text][image5]

2. 
function apply_threshold()

To improve the smoothness of the detection across consecutive frames, I append the function to average consecutive frames (e.g. consecutive 15 frames). 


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

