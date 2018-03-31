##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HogExample.png
[image2]: ./output_images/Features.png
[image3]: ./output_images/SVMcm.png
[image4]: ./output_images/MLPcm.png
[image5]: ./output_images/Windows.png
[image6]: ./output_images/Heatmaps.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 3rd through the 7th code cell of the IPython notebook.

I read in in all the `vehicle` and `non-vehicle` images.  Below are examples of a non car image and a non cat example.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image1]

####2. Explain how you settled on your final choice of HOG parameters.

I picked parameters from looking at random images and seeing what had the largest differences between random cars and not cars. I chose to use `YUV` color space because in various discussions and articles I have heard that `YUV` is very effective for hog features. I only used the `Y` color channel because I have heard that the `Y` channel is more similar to how human's see the world (via rods not cones) than grayscale is. The hog hyperparameters I used are default values I heard about from the Udacity lectures and in discussions from other libraries like dlib. For color histogram and spatially binned features I used 32, again it came from information in the lectures.
In the end my hyperparameters were as follows:

 * Color space = YUV
 * Hog Channel = 0 (Y)
 * orientations = 9
 * pixels_per_cell = (8, 8)
 * cells_per_block = (2, 2)
 * color_hist_bins = 32
 * bin_spatial_size = (32, 32)

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Model selection can be found in the 10th -- 18th cell of the IPython notebook.

To create the dataset I loaded in all the images and calculated features in the form [hog_features:color_hist_features:bin_spatial_features] and they were then scaled. They are then split into a test and validation dataset using sklearn's `train_test_split`. 80 percent of the data was used for training and the remaining 20 was testing data. These features were scaled and an example of what the does is shown here.

![alt text][image2]


I trained a LinearSVM, a LogisticRegressionClassifier, and a "multilayer perceptron" classifier from sklearn which is really a multilayer neural network. The Linear and LogisiticRegression Classifier had almost the same performance (which makes sense a LinearSVM is basically just LogisticRegression with a different (hinge) loss function) so I will only compare the SVM to the MLP. The SVM worked well but I figured that the non-linerarity of the MLP would boost performance. Not only that, but the MLP has multiple layers so it could learn extra representations from the input so I had to do less feature engineering to get good results. The was slightly slower than the SVM but had higher accuracy so I decided to use the MLP. The accuracy of the classifiers were as follows.

```
|       Classifier   | Train Accuracy | Test Accuracy | Train Time | Test Time |
|--------------------|----------------|---------------|------------|-----------|
|Logistic Regression |       1.0      |      0.987    |   25.498   |   0.0002  |
|LinearSVM           |       1.0      |      0.984    |   13.682   |   0.0002  |
| MLP                |       1.0      |      0.99     |   34.86    |   0.0007  |
```

I also plotted confusion matrices for both the SVM and the MLP. There are seen here:

![alt text][image3]

![alt text][image4]

As you can see the MLP not only had better accuracy but it had vastly fewer false negatives when there was a car but it classified it as not a car. This is the most important failure class. If there is not a car and it predicts a car it will just try to avoid that area which should not cause a huge problem (unless it thinks there is a car right in front of it an it swerves hard). However, not detecting a car could easily lead to a accident.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented sliding window search in the 20th cell of the notebook. This was the same implementation that I used to pass the udacity quiz. You can see an example below. I created these windows at a few different size, (64, 96, 128, 192, 256]. These were chosen using the test images. I had to add smaller and smaller scales and more the window less far at each step to get the small car in test image 3. The larger scales are thrown in because they don't add many boxes and should be able to grab cars that are right in front of this, unfortunately there weren't images to test this. To speed this up I used my domain knowledge, i.e. cars are on the road not the sky to only search parts of the image that should be road way where cars can be.

![alt text][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is an example of the classifier.

![alt text][image6]

In order to filter out false positives I used a heatmap based approach. When a car is detected the box is added to the heat map. If the detection is a real car then there will probably be a lot of similar detections where as a false positive will often only have a single detection near it. This means that a simple threshold should get rid of extra detections while keeping real ones. Another solution is to use non-maximum suppression, how ever this just keeps the biggest box so I though an approach like heatmapping would get better boxes. Once I had the head maps I used scipy.ndimage.measurements.label to find the individual cars. This happens in the 23rd cell.

To optimize the performance the first thing I wanted to do was to create a deep neural network that could find all the cars at once, unfortunately due to family illness and a new job I wasn't able to devote enough time to this project as I would have liked. The next thing I tried was to take the HOG features of the whole image at once then subsample windows from that array. This lead to a lot more false positives and I wasn't able to get it to work as well as I would have wanted. Rather than expanding on this I decided to parallelize the window search. The problem is embarrassingly parallel so I divided the windows to search between a few processes. This resulted in some speed up even with my super naive implementation. With more work parallelism will allow for HOG based detection to be incredibly fast, this is showed by the implementation in dlib. dlib also has GPU acceleration so using CUDA would allow for super fast, real time detection. This isn't far fetched, there are already cars on the market that have GPU's on board to run graphic displays.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To smooth over time and filter false positives I used a double ended queue and openCV's groupRectangles function. By adding detected boxes (that are already condensed via a heatmap) to the double ended queue (and old boxes get pushed out) we get a few detections from frame to frame. groupRectangles then combines similar rectangles. A false positive doesn't have any similar rectangles so it is discarded.


In this video you can see the good results (in red) compared to the raw boxes (blue) and the green boxes which don't used the groupRectangles the same way (in fact they are found with openCV's contours). As you can see the groupRectangles both smooth out the blue boxes and removes the false positives which are especially obvious when the black sedan comes into frame.

Here's a [link to bad video results](./all_result.mp4)


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issues I ran into was the HOG of the image and subsampling function. I wasn't able to filter out the false positives which was too bad. Currently speed is an issue but as discussed before effective parallelization will provide great speed up.

In a more general note I find it hard to work with jupyter notebooks because I forget what variables are in scope and the like.