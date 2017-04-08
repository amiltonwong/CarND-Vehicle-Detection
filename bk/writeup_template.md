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
[image5.5]: ./output_images/heatmap_threshold.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/examples_det2.png
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
1. add heatmap to discriminate the true positves and false postives and filter out latter. 

The correspondong part is located in function add_heat() in `cell 16`, which adds "heat" (+=1) for all pixels within windows where a positive detection is reported by my classifier. The individual heat-maps for the above six images look like the following figure:

![alt text][image5]

We could observe that the true postives located in "hot" area in heatmap. The false positives are located in "cool" area. Thus, we could filter out false positives by simply thresholding those "cool" area.

2. impose a threshold to reject areas affected by false positives.

function apply_threshold() in `cell 18` lists the thresholding function. I choose value 3 for threshold parameter. The following figure is thresholded heatmap, we could see the false positives are removed and overlapping detection are reduced.

![alt text][image5.5]

Then, label() funtion is applied to obtain labels image.
Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image6]

Then, put bounding boxes around the labeled regions. The corresponding code is located in `cell 21`.
The following figure shows the improved detection result. We could see the problems of overlapping detection 
and false positives are solved.

![alt text][image7]

The entire pipeline for image detection (with heatmap thresholding) is located in `cell 22`.




**5. Video Implementation**

Here's the [link](https://youtu.be/jJW2VLUC6vI) for the detection output (./project_video.mp4)

For the detection in video, it is easy to have detection suffering jittery bounding boxes. To improve the detection smoothness of the detection across consecutive frames, I append the function to average consecutive frames (e.g. consecutive 20 frames). I choose to average the heatmaps from the recent 20 frames for each new input frame from video: create a class to store data from video, with member variable to store the detection in previous 20 frames. Rather than using a constant scalar in thresholding, I choose a adapative value computed from previous detection history. The corresponding code are located in `cell 24` and `line 59-66 in cell 25`.

---

**6. Discussion**

The main problems in video detection are the false positives problem and temporal coherrence problem. For false positives problem, method such as heat-map thresholding can mitigate it well as implemented in this project. For temporal coherrence problem, one simple and effective way is to consider the average over the accumulated detections in previous several consecutive frames as implemented in this project. To some extent, the smoothness is improved. 
In the entire video detection (project_video.mp4), my method perform reasonably well on the entire project video, the vehicles are identified most of the time with minimal false positives (though somewhat wobbly or unstable bounding boxes come out). For further removing the false positives, some method such as integrating the road lane detection to filter out the "true" detected vechile in opposite road, this method is deserved for further works. 
Also, for vehicle detection, it's worthwile to explore much more powerful discriminative detection methods such as CNN-based methods (SSD/Faster RCNN/YOLO).
