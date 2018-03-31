#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/raw.png "Visualization"
[image2]: ./examples/Extended_histogram.png "Unbalanced"
[image3]: ./examples/example.png "Example"
[image4]: ./examples/example1.png "Jitter"
[image5]: ./examples/Balanced_Histogram.png "Balanced"
[image6]: ./examples/Processed.png "Processed"
[image7]: ./examples/new_sign1.png "Traffic Sign 1"
[image8]: ./examples/new_sign2.png "Traffic Sign 2"
[image9]: ./examples/new_sign3.png "Traffic Sign 3"
[image10]: ./examples/new_sign4.png "Traffic Sign 4"
[image11]: ./examples/new_sign5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.

I used both the numpy and pandas libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the forth and fifth code cell of the IPython notebook.

In the notebook there are several rows showing several examples from each class. One of them looks like this

![alt text][image1]

Further analysis shows the unbalanced nature of the dataset. Some classes are far more common than others and this is exacerbated when augmenting the dataset. Only some signs can be rotated or flipped while retaining meaning. The uneven distribution can be seen here.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 11th to 14th code cell of the IPython notebook.

To fix the unbalanced distribution problem (which is a problem because a model can be biased to guessing at the most common class because it is most likely to be right) I created fake images in the dataset. Creating random perturbed images that are rotated, moved, stretched slightly as well as changing some brightnesses slightly at random created new images to train on while not changing the important features needed for classification. Test and Validation data were not jittered because the point of those datasets is to test the networks ability to work with real world data. These data augmentations allow us to balance the dataset. Here are examples of an image and the types of changes that can be made to it.

![alt text][image3]

![alt text][image4]

These changes help us balance the dataset.

![alt text][image5]

One important thing to note is that these random changes will end up changing the true mean and standard deviation  of the dataset so those statistics are recorded before making these changes. The mean and standard deviation are used to zero the mean of the images and then to normalize the data. The main point of normalizing is to bring features that are of vastly different scales (ie square feet on a house and the number of bathrooms will be features that are super different) into a comparable (0 to 1) range. This is not super needed because all the pixels are already in the same range (0 to 255). However, having large numbers like 255 will make training harder and could cause numerical stability so scaling it down helps training. Examples of processed images are included.

![alt text][image6]

As a first step, I decided to convert the images to grayscale because there are no signs that were the same except for color (this makes sense because some of the human population is color blind and it would not be wise to have drivers need to see colors to know what is going on.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is not present in the IPython notebook.

To cross validate my model, I used the included validation dataset.

My final training set had 650160 number of images. My validation set and test set had 4410 and 12630 number of images.

Generating extra datapoints was discussed in the last section and images are included too.

Here is an example of an original image and an augmented image:

![alt text][image1]

![alt text][image6]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 15th and 16th cell of the ipython notebook.

My final model consisted of the following layers:

```
| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x16 	|
| Batch Norm			| 												|
| ReLU	            	|                                				|
| Max Pooling    	    | 2x2 stride, 2x2 area, outputs 16x16x16		|
| Convolution 5x5		| 1x1 stride, same padding, outputs 16x16x32	|
| Batch Norm	    	|             									|
| ReLU					|												|
| Max Pooling			| 2x2 stride, 2x2 area, outputs 8x8x32			|
| Convolution 5x5		| 1x1 stride, same padding, outputs 8x8x64		|
| Batch Norm 			| 												|
| ReLU 					|												|
| Convolution 5x5 		| 1x1 stride, Valid padding, outputs 4x4x128 	|
| Batch Norm 			| 												|
| Flatten				| Flattens to 1D. outputs 2048					|
| Fully Connected		| Outputs 4096, regularized via weight decay    |
| ReLU 					|												|
| Fully Connected 		| Outputs 4096, regularized via weight decay    |
| ReLU 					|												|
| Fully Connected       | Outputs 4096, regularized via weight decay    |
| Logits 				| Outputs 43, one for each class. 				|
| Softmax 				| Output prediction probabilities.              |
|                       | This is not used for training only prediction.|
```


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 20th cell of the ipython notebook.

To train the model, I used a convolutional neural network. Convolutional networks are very good at finding local patterns regardless of where they appear in the image. Not only are they theoretically well suited to these sort of image classification but thoughout history they have been used to great success, from the original LeNet to all the ConvNets that have won Imagenet competitions (AlexNet, ZFNet, VGGnet, GoogLeNet, ResNet). I trained my network using batch normalization. In the literature batch norm has been shown to help achieve greater accuracy so it makes sense to use it. What's even better is the Batch Norm often speeds convergence which makes it a huge boon for doing quick experiments and allowed me to iterate quickly. I used weight decay to regularize my network rather than dropout because Batch Norm also has a regularizing effect due to it's reduction in internal covariate shift. Originally I had no regularization but there was signs of overfitting so I added weight decay. Large batchsizes (256) was used because the images are small, I have a high end GPU and Batch Norm works better with large batch sizes so that the mini-batch statistics are a closer approximation of the who training dataset statistics. I used the Adam Optimizer because it has in my experience given a lot of really good results without much work. The default values were used and a learning rate of 1e-3 was chosen based on previous experience and in practice it seemed to work well. 10 epochs were used because it was a nice balance between convergence (going beyond 10 didn't help much) and speed (10 epochs could be trained quickly).

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 19th cell of the Ipython notebook.

My final model results were:

* training set accuracy of 98.2%
* validation set accuracy of 97.7%
* test set accuracy of 95.7%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
    * The first architecture I used was LeNet. It was chosen because it is the classic image recognition network.
* What were some problems with the initial architecture?
    * This model had both high training error and high validation error might that it was not powerful enough to solve the problem.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    From LeNet the model was adapted to a hybrid of AlexNet (the multiple layers of 4096 fully connected layers) and VGGNet with repeated applications of small convolutions. However this network had much higher validation error than training error so regularization in the form of Batch Norm and weight decay was added which brought the errors closer together, in some training runs the validation error was actually less than the training error, this is because the training set has a lot of hard, fake data so better performance on the validation set means that the network was learning the latent features that really define the signs. Analyzing errors I found that a lot of errors were signs like Road work and Children crossing which had the same basic shape (triangle with red outlines) that had different interiors. I thought this might be due to the small size of the convolutional features so I upped the first one to 11 like the original AlexNet and it helped fix these errors.
* Which parameters were tuned? How were they adjusted and why?
    * Parameters were tuned by looking at how the validation accuracy was compared to the training accuracy and decisions were kept that brought validation up while not hurting the training accuracy too much (over regularization for example).
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    Important design choices like Batch norm and weight decay were discussed in the previous section.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image11]

![alt text][image9]

![alt text][image7]

![alt text][image10]

![alt text][image8]

The 2rd image could be hard to classify because 'Road work' and 'Children crossing' are both triangles with a red outline and a lot of black within the triangle compared to the amount of black in signs like 'General caution'.

The 4th image is also pretty had because it is not a straight on view. Plus it also is less circle looking when it is scaled down to 32x32 as seen in the notebook.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

```
| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Yield          		| Yield      									|
| Road work    			| Road work     								|
| Speed limit (70km/h)  | Speed limit (70km/h)							|
| Speed limit (100km/h)	| Stop      					 				|
| Priority road			| Priority road      							|
```

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares is quite a bit lower than the accuracy on the test set which was 95.4%. This is dis-hearting but understandable because I got these images from the Internet. This means that the underlying distribution compared to the training dataset is very different. The train, validation, and test data are all from the KITTI dataset with the same data collection systems this means that patterns from the training dataset are more likely to be in the test dataset.

Some of these images are also from different angles than the training data images. This can hurt the classification and leads to this poor accuracy but would be a less prevalent problem in a real car because the point of view in cars would be consistent.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

In all correct cases the model is super sure in its prediction. When it is wrong it is still very sure of itself but less so than when it is correct which is a good sign.

Fist Image was a Yield Sign.
```
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .998      			| Yield     									|
| .00085  				| Keep right									|
| .00033				| Speed limit (50km/h)							|
| .000085     			| Keep left 					 				|
| .000084   		    | No vehicles                       			|
```
Second Image was a Road Work sign.
```
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .957         			| Road work     								|
| .025     				| Double curve 									|
| .008					| Road narrows on the right						|
| .003	      			| Beware of ice/snow                			|
| .001				    | Pedestrians        							|
```
Third Sign was a Speed limit (70km/h).
```
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .099         			| Speed limit (70km/h)							|
| .0006648   			| Speed limit (20km/h)							|
| .000146    			| Speed limit (30km/h)							|
| .0000043     			| Stop                   		 				|
| .0000011			    | Roundabout mandatory 							|
```
Fourth image was a Speed limit (100km/h) sign.

This is the most wrong prediction that the network makes but there are some speed limit signs in the predicitons which is good. It as also good that the network is (slightly) less confident in it's answer. Also in a real world scenario missing a speed limit sign would be bad if the speed limit was changing but that would only result in a ticket rather than missing a sign like lanes merging an accident could be caused.
```
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .079         			| Stop   							     		|
| .019     				| No vehicles           						|
| .003					| Speed limit (70km/h)							|
| .0011      			| Speed limit (50km/h)			 				|
| .0003				    | Keep right       					|
```
The last image was a Priority Road sign.
```
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Priority road									|
| 9e-10    				| Roundabout mandatory							|
| 3e-10					| No entry										|
| 6e-11	      			| Speed limit (30km/h)			 				|
| 5e-11				    | Keep left         							|
```