#**Finding Lane Lines on the Road** 

[//]: # (Image References)

[image1]: ./test_images/solidWhiteCurve.jpg
[image2]: ./output/solidWhiteCurve-gray.png
[image3]: ./output/solidWhiteCurve-blurred.png
[image4]: ./output/solidWhiteCurve-canny.png
[image5]: ./output/solidWhiteCurve-masked.png
[image6]: ./output/solidWhiteCurve-hough.png
[image7]: ./output/solidWhiteCurve-final.png

---

### Reflection

My pipeline involves 6 steps. Using this picture I will show an example of 
the steps on this image.

![alt text][image1]

 * Step 1. The first step is converting the image to grayscale. This is done
 because the color of the line line, while important when you trying to make 
 decisions based on the lines, is not needed for detecting them. Only the 
 intensity of the lines matters.

 ![alt text][image2]
 
 * Step 2. The second step it to blur the image. The gradient between a line 
 line and the road is often much stronger that other gradients so blurring large 
 area is much more likely to get rid of extraneous gradients than the important 
 ones. Large blurs is also acceptable because lane lines are often far from anything 
 else in the picture besides the road so they will not be affected much. The 
 Gaussian kernel cannot be too big however because lane lines close the the edge 
 of the road are often much closer to  things that will mess up its gradient.
 Gaussian blurring was done with a kernel of size 5.
 
 ![alt text][image3]

 * Step 3. The third step is to use Canny edge detection to find the edges in 
 the image. This is controlled by two thresholds. The lane lines are white and most 
 of the road is very dark so the gradient is very strong. This means that by choosing
 a large threshold we can ignore edges that are created by other things in the 
 scene. The threshold cannot be too large because the gradient between 
 yellow lane lines and the road is smaller than the gradient between white lines
 and the road. The numbers used was a high_threshold of 150 and a low_threshold 
 of 50 (this follows the 1:3 ratio suggested by Canny) 

 ![alt text][image4]

 * Step 4. The fourth step is to mask out areas that lane lines are not likely 
 to appear in. I used a polygon defined by the following points 
 ([0, image height],[465, 320],[475, 320],[image height, image width]). This 
 box was created experimentally by using various sizes on various images.

 ![alt text][image5]

 * Step 5. Step 5 was to use a Hough Transformation to create lines from the Canny
 edge detections. The nature of lane lines helped set parameters for the Hough 
 transformation. Lane lines should have a lot of Canny edge dots so the threshold 
 can be large. Lane lines are also rather large so the minimum length can be 
 rather large. However it can't be too large because segmented lane line cannot 
 be over looked. A large max gap is also important because it allows us to connect 
 segmented lane lines. The values used where rho = 2, theta = pi/180, threshold 
 = 45, min_length = 40, and max_gap = 100.

 ![alt text][image6]

 * Step 6. The final step was to create a single line from the line returned from 
 the Hough transformation. This was done by dividing the lines based on their 
 slope. Positive slope means the line is most likely the left lane line and a 
 negative slope means it is probably the right lane line. Once the lines are 
 divided the average line was found based on their slopes. Slopes that were very 
 far from the mean (more than 1.5 times the standard deviation) are removed 
 because a very different slope means it is most likely an edge from something 
 other than a lane line.  

 ![alt text][image7]

 * Finally these lines are super imposed on the input image.

###2. Potential shortcomings in the current pipeline

A problem with my pipeline is how the final lane is calculated. I only use the 
slope to get rid of outliers and not the intercept. This was not a problem in 
the test data but it could be. For example, if the edge formed from the edge of 
the road was detected. This would create a line with a similar slope to the lane 
line but a very different intercept. This would pull the average line towards 
the edge of the road away from the lane line. 

Another shortcoming is what would happen if the road is very light. New concrete 
is a much lighter shade of gray than normal road. The gradient thresholds are set 
rather high to help reduce the noise in the detected edge images but when the 
lane line gradients are weaker due to the lighter road sometimes not edges are 
found. 

A third problem is that the system cannot handle curves. When the road curves 
the segments of Hough lines would start with a negative slope and change to a 
positive slope. This would group segments that are on opposite sides of the 
road and draw lines that are in the middle of the road and they would cross. 
A curve in the road does not follow the assumptions used to optimize the pipeline
for straight lines so it presents a problem for the system.

There are several problems with this pipeline that become apparent when more 
complex situations are encountered.

###3. Possible improvements to the pipeline

The first improvement would be to re-optimize the system to preform better when 
the gradient between the road and lane lines is weaker. With this weaker gradient 
threshold there will be more noise in the image. So this improvement would not 
help much unless other improvements were made. 

The second improvement would be to create better clustering methods. Currently 
the lines are grouped using their slopes. As discussed before this falls apart 
when the lane line is curved. A better clustering algorithm would help solve this.
For example K-means clustering with a k of 2 should be about to separate the points
that form the left and right lane lines pretty well. Another clustering solution 
would be to separate the points based on things like distance from the side of 
the image. This would be done with some thing like a Support Vector Machine (with 
a non-linear kernel because if the curve is shape enough the points might not 
be linearly separable). Better clustering would help handle curves much better.

Another improvement would be to have a better fitting algorithm. This goes hand 
in hand with a better clustering algorithm. As the system is able to split more 
complex patterns into groups a more complex line needs to be draw. My first 
thought was linear regression which would help draw better lines for the current 
pipeline. However this would not help for curves because the line would not be 
linear. A more complex line must be fit. Luckily the most complex curve a line 
line could be could most likely be fit by a quadratic function so not much 
regularization is needed.

There are a lot of improvements to be made to the system to handle the more 
complex situations found in driving.