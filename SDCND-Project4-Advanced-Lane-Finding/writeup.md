# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Chessboard_points.png "Points"
[image2]: ./output_images/undistort_example.png "Undistort"
[image3]: ./output_images/undistort_road.png "Road"
[image4]: ./output_images/undistort_road2.png "Road 2"
[image5]: ./output_images/warp_image.png "Transformation"
[image6]: ./output_images/threshold_example.png "Thresholding"
[image7]: ./output_images/lane_example.png "Found Lanes"
[image8]: ./output_images/Lane_on_image.png "Example"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd code cell of the IPython notebook located in "./Advanced Lane finding.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The sample writeup is exactly what I did, just like in the camera calibration lab. Here is a chessboard with detected corners on it.

![alt text][image1]

The calibration was done in the 3rd code cell. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  This returned the mtx and dist camera matrices which were cached to use later. The undistortion was done with `cv2.undistort()` function and demoed in the 4th code cell. Here is an example of an unditorted chessboard image.

![alt text][image2]

I also looked at the undistorted images from the test_images folders. Here is an example.

![alt text][image3]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one and the last one:

![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for the thresholding was in the 8th code cell.

The main things I used to threshold was color thresholding and gradient thresholding. For color thresholding I used the S channel from HLS color space which from experimenting picked up Yellow lines really well. I also used the b channel from the Lab colorspace which also helped pick up yellow lines. I used the L channel from the LUV which found white pretty well. I also used gradient thresholding via magnitude thresholding. I also tried to use direction thresholding but in the end it came out as too noise and added a lot of bright pixels to the middle of the lane. The different binary images can be seen below.

![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the 6th code cell. I calculated `src` and `dst` points in the `calculate_warp_points` function. I drew the points on the original image to show how it would work pretty well. There are also functions to find both the transformation matrix M and the inverse transformation matrix Minv in the `perspective_matrices()` and the function `perspective_warp()` does the actual transformations. I decided to transform the image into a portrait style so the line are longer and easier to find. An example shown here. In this implementation I decided to switch the image dimensions when doing a perspective transformation. This allowed for the view of the lanes to be longer and skinnier like a lane itself. I thought that this would help detection.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the 9th code cell I found lane lines. To start I found the histogram of the bottom part of the images. The location of the peaks is most likely the lane lines so I start my search there. I create a box around the peak and then start to search the image above the current box looking for the part with the most white pixels. This box is then used to start the search at the next height. When no box is found the search just starts at the side sweeps over. This allows for gaps in the boxes that are common due to the broken white lines. Then I fit the positions to polynomials with `np.polyfit()` based on both the pixels and based on the boxes. You can see an examples here along with the histogram in blues and peaks marked with red x's

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In the 11th code block I calculate the curvature. I simpled based it on the formula in the lectures with `poly_cr[0]` as A and `poly_cr[1]` as B. Conversions of pixels to meters is based on some discussion and values I saw on the udacity forums. As you can see the curvature hovers around 1,000 m which is very close to the 1 km discussed in the tips page. The position of the car is found in the `x+center_offset()` in the Lane class this is done by finding the location of the center of the two detected lanes and comparing that to the center of the image in the horizontal axis. In this case a positive distance means the car is toward the left side of the lane.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 12 through 14 in notebook. This code is some classes to implement to the frame by frame smoothing and recovery from bad detections. Also in the 14th code block I draw the lane in the function `_draw_lanes_unwraped()` in the Lane class.

Here is an example image taken from the video.

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I used some classes to hold the previous detected lane lines and use previous detections to smooth and recover from bad detections.

Here's a [link to my video result](./project_video_with_lines.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the largest problems I faced was implementing the frame by frame techniques. I already had a lot of experience in computer vision and machine learning so the things I struggled with the most was implementing good temporal processing. I think discussion of this is a lecture would have been nice because I feel that the video stuff is super important, it lets us predict things in the future, but is probably the weakest part of the project.

My pipeline can fail in a couple of places. The first is when there is a huge turn in the raod. In that can the lanes quickly turn off to the side and a large amount of the transformed space is wasted. If the turn is sharp enough then the right lane could even be above the left one and conceivably be detected as part of the left line.

The other spot it could fail is in shadows. Even though I worked a lot on detecting yellow lines regardless of light it is still easy to have problems, for example in the fourth test image in the thresholding pictures there is a lot of cruft on the left side of the image that I think is due to the shadow. In a similar problem white lines are not as well detected at the yellow lines because I put a lot more effort into finding the yellow because it was missed often in the first lane line project. Both of these problems can be addressed by more feature engineering, i.e. finding better thresholds, finding better colorspaces, and combining features in smarter ways.