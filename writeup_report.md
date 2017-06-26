# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./documentation_img/network_architecture.jpg "Model Visualization"
[image2]: ./documentation_img/udacity_data.png "Udacity data image example"
[image3]: ./documentation_img/problem_area.png "Problem area data example"
[image4]: ./documentation_img/throttle_initial.png "Initial throttle histogram"
[image5]: ./documentation_img/throttle_afterwards.png "After preprocessing throttle histogram"
[image6]: ./documentation_img/steering_initial.png "Initial steering angle historgram"
[image7]: ./documentation_img/steering_afterwards.png "After preprocessing steering angle histogram"
[image8]: ./documentation_img/camera_positions.png "Camera position examples"
[image9]: ./documentation_img/cropping_before.png "Example image before cropping"
[image10]: ./documentation_img/cropping_after.png "Example image after cropping"
[image11]: ./documentation_img/resizing_after.png "Example image after resizing"
[image12]: ./documentation_img/image_flip.png "Fipping image example"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json containing the convolutional neural network model
* model.h5 containing the trained weights of the convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

#### 3. Submssion code is usable and readable

`model.py` contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model is based on Nvidias [best practice model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which I enhanced by dropout. I experimented with various model structures but eventually the Nvidia structure fitted my problem best.

My model consists of a convolution neural network with filters of varying strides (`model.py` line 85 ff.), kernel sizes (`model.py` line 92 ff.) and depths between 24 and 64 (`model.py` line 78 ff.). The model includes RELU layers to introduce nonlinearity (`model.py` line 127 ff.). The data is normalized in the model using a Keras lambda layer (code line 126). The detailed model structure will be provided in section "Model Architecture and Training Strategy" in point "2. Final Model Architecture".
   


#### 2. Attempts to reduce overfitting in the model

* Normalization with Keras lambda layer (`model.py` line 126)
* Dropout layer after the first convolutional layer (`model.py` line 128)
* Duplicated images with a resulting steering input of >0.1 by applying flip to image and revere to corresponding steering angle to prevent overfitting on data with small steering angles (vast majority of test data) (`model.py` line 190 f.)
* Reshuffling training data after all images have been seen (`model.py` line 269)
* Random choice of image position (left, center, right) during batch generation (`model.py` line 273 ff.)
* Flip images and reverse steering angle with a probability of 0.6 during batch generation to prevent overfitting on data biased to the left (as training course takes mostly left turns) (`model.py` line 290 f.)


The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 150).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I started with the basic Udacity training data and added additional training data myself by driving with the beta simulator providing smooth steering angles by mouse inputs. In the end I duplicated the data for parts of the track, which seemed underrepresented to me due to their specific nature. 

For details about how I created the training data, see section "Model Architecture and Training Strategy" in point "3. Creation of the Training Set & Training Process". 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was initially an experimental one. I started with one convolutional and one fully connected layer and increased the number of layers. However, this model was eventually not able to keep the car on the track for the full round.
Consequently, I looked into existing best practices and created a model similar to the one of the [Nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
I thought this model was an appropriate starting point as it already proved its qualities in the real world driving scenario and therefore might also be suitable to the simulator driving scenario.

In order to gauge how well my models was working, I split my image and steering angle data into a training and validation set already while experimenting with my own model structure. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

In order to prevent overfitting I added a dropout layers to my model. For my final architecture I just kept one of them after the first convolution as this yielded the best results. Furthermore I made the model choosing the image position (left, center, right) randomly during batch generation, while adding a corrector for the steering angle bias of left and right images.

Both testing with my own model structure and with the one inspired by Nvidia, the car was driving around the track quite well in the simulator. However at some spots the car was struggeling. To overcome this challenge I duplicated the data for parts of the track which seemed underrepresented in the dataset to me in order to show these track pieces more often to the model.

Furthermore, I realized, that most of the data had very small steering angles. In order to get a more balanced datastructure, I duplicated images with a resulting steering input of >0.1 by applying flipping and reversing of the steering input in order to prevent overfitting on data with small steering angles.

In the end the car was just touching the curbs at the right turn of the track. In order to prevent this issue which I think was caused due to overfitting on the mostly left turns of the track I introduced random flipping with a probability of 0.6 to images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road or hitting the curbs at full speed.

#### 2. Final Model Architecture

The final model architecture (`model.py` lines 123 ff.) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I initially started with the training data provided by Udacity.

Example: 

![alt text][image2]

However, I realized that I needed more training data in order to make suitable prediction with my model. I recorded several own rounds on the training track to capture further good driving behavior. For critical parts in the track (e.g. sections which display unique patterns for a very short section in the track) I created extra data by driving and duplicating the collected data in order to create a balanced datastructure representing all patterns of a track. Overall I thereby ended up with a dataset of 17841 datapoints

Example for crutial area where I created extra data:

![alt text][image3]

In the data preprocessing pipline I removed training data with a throttle of less than 0.5 as the car eventually would drive around the track at full speed (`model.py` line 19 f.).

Throttle histogram before:

![alt text][image4]

Throttle histogram afterwards:

![alt text][image5]

I also realized that there was very little data for strong steering angles:

![alt text][image6]

To address this problem, which might result to a model bias towards choosing low steering angles, I duplicated the data for steering angles greater than 0.1, by flipping the image and reversing the steering angle (`model.py` line 22 ff.). Afterwards the data distribution looked like this:

![alt text][image7]

This additional creation of data increased the datapoints in the dataset to 18237. After a shuffle to randomize the data (`model.py` line 38 f.), I splitted the data into a training and a validation set with a split of 9:1 (`model.py` line 41 f.). The train set size was 16417, while the validation set size was 1825. The validation set helped determine if the model was over or under fitting.

In order to also address the issue regarding the lack of data for strong steering angles and additionally the lack of recovery data, I decided to not only use the center camera images, but also the left and right ones. During batch generation they are randomly selected. If a left or right image is chosen, the steering angle is corrected by 0.1 for left and -0.1 for right images in order to achieve convergence of the car towards the middle of the track (`model.py` line 178 ff.). 

Example of the three image positions for 1 datapoint:

![alt text][image8]

The horizion in the image does not provide information about the edges of the road. The same holds true for the foreground displaying the front of the car. Consequently, I decided to crop the image top and bottom in the preprocessing pipeline (`model.py` line 283 f.).

Example image before cropping:

![alt text][image9]

Example image after cropping:

![alt text][image10]

Subsequently, I resized the image to 66x200x3 in order to provide the right input structure for the Nvidia architecture (`model.py` line 286 f.):

![alt text][image11]

In the steering angle histograms seen before, one can also see, that the distribution of steering angles is left centered. Consequently the model will probably overfit towards left turns. To address this issue I implemented a random image flip with steering angle reversion during batch generation with a probability of 0.6, to show the model a more balanced representation of left and right turns (`model.py` line 289 ff.).

![alt text][image12]

After all datapoints have been seen by the batch generator it automatically reshuffles the data (`model.py` line 268 ff.).

I used the training data for training the model. The validation set helped determine if the model was over or under fitting. Through empiric testing I determined that the ideal number of epochs for my model was 5 with a batch size of 64. To determine the loss I used the Minimum Squared Errors method. I employed an adam optimizer so that manually training the learning rate wasn't necessary.
