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
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
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

2. Parameters for HoG feature and comparison
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

The corresponding part in parameters configuration is listed in `cell `.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

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

