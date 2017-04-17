#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goal of the project was to use a car driving simulation to train a neural network how to steer, based on a camera images attached to the car.
This required to:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Model Architecture and Training Strategy

### 1. Solution Design Approach

My first step was to gather some data so I could start designing an architecture. I simply recorded a couple laps on Track1 to start training a model.

I've tested multiple network architectures for this project. I was still getting familiarized with Keras, so my first step was trying to start with a very simple network, like LeNet. With LeNet, I got poor performance. Although the car was able to keep driving straight and steer on most curves, it didn't show a very natural behavior, and would easily get off road without being able to recover.

After that, I decided to try the network architecture from the NVIDIA [End to End Learning for Self-Driving Cars Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This architecture gave me results in comparison with LeNet. I've also tried a few different hyper-parameters, and adding dropout layers in between the existing layers, which seems to have improved slightly some overfitting.

I then decided to try feature extraction with some known architecture packaged with Keras. The one I got the best result was VGG, but it felt like the NVIDIA model did better, so I used it mostly to practice the concept of feature extraction and went back to the NVIDIA insipired architecture.

I added dropout layers between the convolution layers. I reached with value of 20%. Less than that, the car seems to weave too much between both sides of the road. Higher values makes the car keep driving straigh with very small angles.

I used the Adam optimizer with a mean squared error function. I also experimented with the mean absolute error, but had no improvement, so I kept with MAE.

### 2. Final Model Architecture

My final architecture was a slightly modified NVIDIA architecture. The modifications were the following:

- 160wx80hx3c input data
- A 20% dropout layer between each convolution to reduce overfitting, mainly for the constant straight driving
- I removed the first fully connected node after the convolutions, which was a 1164 nodes. That due to the much smaller input size compared to the original network.

#### Final Architecture:

- Layer 1: Lambda function to normalize values to a 0 mean.
- Layer 2: Convolution Layer (5x5 filter with 24 depth, and 2,2 max pool, ReLU activation)
- Layer 3: 20% dropout
- Layer 4: Convolution Layer (5x5 filter with 36 depth, and 2,2 max pool, ReLU activation)
- Layer 5: 20% dropout
- Layer 6: Convolution Layer (5x5 filter with 48 depth, and 2,2 max pool, ReLU activation)
- Layer 7: 20% dropout
- Layer 8: Convolution Layer (3x3 filter with 64 depth, ReLU activation)
- Layer 9: 20% dropout
- Layer 10: Convolution Layer (3x3 filter with 64 depth, ReLU activation)
- Layer 11: 20% dropout
- Layer 12: Fully connect 100 nodes layer
- Layer 13: Fully connect 50 nodes layer
- Layer 14: Fully connect 10 nodes layer
- Layer 15: Output 1 node layer

#### Hyper parameters and details

- Batch Size: 64
- Epochs: 5
- Validation/Training Split: 70%/30%
- Adam optimizer, with mean squared error loss function.


### 3. Creation of the Training Set & Training Process

When I started collecting training data, I wasn't doing it very strategically, I was just driving around and recording it. After I started working more on the model creation, I realized that I had very little visibility of how much of my data was for each type of training I should have (driving straight, curve, recovery, bridge, etc.).  Was also virtually imposible for me to remove specific parts of the training set that might be affecting the model negatively. To solve that, I modified my code to instead of looking into a main .csv file for the data, I would look for all .csv files in a target folder. I then started a new training data, and I would rename the .csv for each specific scenario and driving behavior I was training. 
That approach was crucial to get a good training data, as I soon realized that some data did not improve the model, and others even made it worst. I could mix and match any combination of data sets for the trainig.

I ended up creating different data sets for the following types of driving:

- Driving correctly and trying to stay as centered on the lane as possible.
- Driving well, but being more conservative on forcing the car to the center and trying to have more 0 angle steering
- Recovering from off-center of the lane
- Recovering from out of the lane
- All of the above, but driving on different direction

My final selected training data has 22268 data points.

After collecting the data, I also applied some pre-processing to the image, which is also applied to the images the drive.py sends to the model. The following was applied:

- Inspired by the advanced lane finding lessons, I applied a transformation to the image to remove perspective. From a human perspective, was easier to identify looking at the modified images whether the car should be making a turn or keeping centered. I noticed improvements in the model after this transformation.
- I applied an histogram equalization to the Y channel of the image, after converting it to the YUV color space. I converted back to RGB after.
- I applied a Gaussian Blur. That seems to have reduced overfitting, specially on the most pixelated areas of the image.


To augment the data set, I also applied the following:

- I used the cars side cameras, with a steering adjustment towards the center camera of 0.2, but only for steering angles higher than 0.2. I did that so I would augment most straight angles, as they were already too prominent in the data.
- I flipped each image and its angle (including the side camera ones)