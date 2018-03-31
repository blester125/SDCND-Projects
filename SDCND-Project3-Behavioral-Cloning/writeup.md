# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_05_22_23_57_59_655.jpg "Center"
[image2]: ./examples/left_2017_05_22_23_57_59_655.jpg "Left"
[image3]: ./examples/right_2017_05_22_23_57_59_655.jpg "Right"
[image4]: ./examples/YUV_center.png "YUV"
[image5]: ./examples/YUV_cropped.png "Cropped"
[image6]: ./examples/YUV_resized.png "Resize"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model uses a few sizes of convolutional layers (5, 3) at different strides (2, 1) followed by several fully connected layers. The model uses ReLU layers to introduce non-linearities needed to learn non-linear functions. The incoming data uses YUV colorspace and is cropped only the road is used. This is done in the `preprocess_image()` function. The images are also resized to 200 pixels wide and 66 pixels tall as used in the NVIDIA paper. The images are then normalized using a Lambda layer in the Keras model. The model also uses Batch Normalization to speed convergence.

####2. Attempts to reduce overfitting in the model

The model fights overfitting via a training and test data split as well as with Batch Normalization layers which have been show to not only speed convergence but also to improve generalization.

The model was trained and validated on different data sets to ensure that the model was not overfitting This split was created with sklearn. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer but a smaller learning rate was used because I read something similar in the udacity slack. Thinking back I probably could have used a much larger due to my use of batch normalization.
####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and left and right camera images. Left and right images were used with a large correction factor. This is evident in the video where the car jitter back and forth due to this over compensation.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build iteratively. 

My first step was to use a simple MLP with a single hidden layer (theoretically this can approximate any function so it a ok start), this was only to test that I could both train the network and make it run in the simulator.The next step was a convolution neural network model similar to the LeNet, this is the graddaddy of all conv nets so it seemed like a good place to start but with obvious modification to handle 3 color channels.I thought this model might be appropriate because I was working with images and conv nets are great at that.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was underfitting so I upped the ante and jumped to the nvidia architecture because they have showed it worked in real life. This model seemed to work but had long training times and was prone to overfitting (training errors much lower than validation errors), so I knew I needed to add regularization.

To combat the overfitting, I modified the model to add Batch Normalization which fixed both of my problems because it is known to both speed convergence and reduce overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track namely the curve with the dirt road after the bridge, the car would drive onto the dirt (this is obviously a tricky part and was put there on purpose). To improve the driving behavior in these cases, I created recovery data where I would be almost on the dirt road and make a big swing to the left to avoid the dirt.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 101-1274) consisted of a convolution neural network with the following layers and layer sizes.

The first 3 convolutional layers had kernels of 5 by 5 with depth 24, 36, 48 respectively and strides of 2 in each direction with valid padding. The next two were both 3 by 3's with a depth of 64 and valid padding. Each layer was followed by Batch Normalization and a relu non-linearity. The output was then flattened and there were 4 affine layers with sizes, 1164, 400, 50, and 10, followed by relu activation. The final layer was a single output layer and mean squared error was used as a loss function.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
batch_normalization_1 (Batch (None, 31, 98, 24)        96        
_________________________________________________________________
activation_1 (Activation)    (None, 31, 98, 24)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 47, 36)        144       
_________________________________________________________________
activation_2 (Activation)    (None, 14, 47, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
batch_normalization_3 (Batch (None, 5, 22, 48)         192       
_________________________________________________________________
activation_3 (Activation)    (None, 5, 22, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
batch_normalization_4 (Batch (None, 3, 20, 64)         256       
_________________________________________________________________
activation_4 (Activation)    (None, 3, 20, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
batch_normalization_5 (Batch (None, 1, 18, 64)         256       
_________________________________________________________________
activation_5 (Activation)    (None, 1, 18, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              1342092   
_________________________________________________________________
activation_6 (Activation)    (None, 1164)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
activation_7 (Activation)    (None, 100)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_8 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_9 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 1,596,455
Trainable params: 1,595,983
Non-trainable params: 472
_________________________________________________________________
```

The lambda layer was used to normalize the images.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then used the vehicle's cameras on the left side and right sides and used a correction factor of 0.25 (added to left image's angle and subtracted from the right image's angle) to simulate steering back to center. In the NVIDIA paper they didn't talk about recovery from the sides like the lectures did so I had not even tried that when I had a model that could drive around the track. My data augmentation is similar to the Nvidia paper.

![alt text][image2]
![alt text][image3]

After the collection process, I had 27405 number of data points. I then preprocessed this data by converting it to YUV (as per the Nvidia paper) seen here:

![alt text][image4]

Then I cropped it to remove the car itself, and the background like so:

![alt text][image5]

I also resized it to be the size used in the Nvidia paper.

![alt text][image6]

I finally normalized the images using a Keras lambda layer. Visualization of this just looks like a black screen in most image views so it has been omitted.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the callback function only saving the best model which often ended up being after the 5th or so epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
