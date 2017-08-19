# **Behavioral Cloning** 

### Writeup by Hannes Bergler
---

**Behavioral Cloning Project**

The goals / steps of this project were the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Driving"
[image2]: ./examples/recovery1.jpg "Recovery Data1"
[image3]: ./examples/recovery2.jpg "Recovery Data2"
[image4]: ./examples/recovery3.jpg "Recovery Data3"
[image5]: ./examples/recovery4.jpg "Recovery Data4"
[image6]: ./examples/recovery5.jpg "Recovery Data5"
[image7]: ./examples/not-flipped.jpg "Image Before Flipping"
[image8]: ./examples/flipped.jpg "Flipped Image"
[image9]: output.PNG "Training Output"


## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### I. Submission Files

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - containing the script to create and train the model
* drive.py - for driving the car in autonomous mode
* model.h5 - containing a trained convolution neural network 
* run1.mp4 - a video of the car driving track one in autonomous mode
* writeup.md - summarizing the project results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### II. Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 73-84).

The model includes RELU layers to introduce nonlinearity (e.g. code line 75), and the data is normalized in the model using a Keras "BatchNormalization" layer (code line 69).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 89, 92, 95). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 101). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also the training data provided by Udacity.

For details about how I created the training data, see the next section. 


### III. Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one I used in the traffic sign classification project. I thought this model might be appropriate because it has already proven to be a decent image classification model.
I found that the model from the traffic sign classification project produced a very large model file (caused by the big fully connected layers) and did not perform very well in the simulator.

Then I switched to the "even more powerful network" by NVIDIA proposed in class. This network has five convolutional layers and three fully connected layers.

To prevent overfitting, I added maxpooling layers to the convolutional layers and dropout to the fully connected layers.

This model performed much better in the sumulator than the first approch described above and also produced smaller model files due to the smaller fully connected layers.

But still there were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected more recovery training data from these critical spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 67-97) consisted of a convolution neural network with the following layers and layer sizes:

```sh
# convolutional layers...
model.add(Convolution2D(24, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Flatten())

# fully connected layers...
model.add(Dense(100))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Dense(50))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded one laps on track one using center lane driving, in addition to the training data provided by Udacity. Here is an example image of center lane driving:

![Training Data1][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it is driving to close to the edge of the track. These images show what a recovery looks like starting from the left side of the track:

![Recovery Data1][image2]
![Recovery Data2][image3]
![Recovery Data3][image4]
![Recovery Data4][image5]
![Recovery Data5][image6]

Then I collected one laps of center lane driving from track two to help generalizing the model.

To augment the data sat, I also flipped images and angles thinking that this would be a good way to overcome the left turn bias from track one. For example, here is an image that has then been flipped:

![Training Data4][image7]
![Training Data5][image8]


After the collection process, I had 35548 data points (also counting flipped images). I then preprocessed this data by cropping of 67 pixels at the top and 25 pixels at the bottom of the images to prevent the model from getting distracted by the environment like trees, lakes, etc.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used five training epochs, the training and validation loss is depicted in image below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

!["Training Output"][image9]
