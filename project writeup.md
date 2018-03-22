# **Traffic Sign Recognition** 

## Writeup

### Yulong Li

  

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize and analyze feature maps
* Summarize the results with a written report

 
---
## Rubric Points
This writeup will include all the [Rubric Points](https://review.udacity.com/#!/rubrics/481/view).  
Here is a link to my [project code](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

---
### Dataset Exploration
#### Dataset Summary
The provided German Traffic Sign Dataset consist 34799 traning images, 4410 validation images and 12630 testing images, which contain 43 different classes. All images are 32x32 3 channel RGB. The data are provided in pickle format, which contains four pairs of key and value: feature (raw pixel data of images), label, image size and traffic sign object coordinates.

#### Exploratory Visualization
Below are three examples from the training dataset:  
![2](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/2.png)
![12](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/12.png)
![26](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/26.png)  

Below are histograms for different classes in training, validation and testing data set:  
![trainhist](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/trainhist.png)
![validhist](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/validhist.png)
![testhist](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/testhist.png)  

---

### Design and Test a Model Architecture

#### Preprocessing

I decided not to convert the images to grayscale because I think color information is useful for traffic sign classification. For example the 'ahead only' sign is blue and the 'stop' sign is red. I want the CNN also learning the color information.  

I normalized the image data because it can help to avoid very big positive or negative values after the convolution and it's easier for optimizer tp work.  

#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3x6     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5x6x16	    | 1x1 stride, VALID padding, outputs 24x24x16	|
| RELU					|												|
| Convolution 5x5x16x26	    | 1x1 stride, VALID padding, outputs 20x20x26	|
| RELU					|												|
| Convolution 5x5x26x36	    | 1x1 stride, VALID padding, outputs 16x16x36	|
| RELU					|												|
| Convolution 5x5x36x36	    | 1x1 stride, VALID padding, outputs 12x12x36	|
| RELU					|												|
| Maxpooling 2x2				  	|	2x2	stride, SAME padding, outputs 6x6x36		|
| Fully connected	1296	| Outputs 666        									|
| RELU					|												|
| Dropout				|	0.75								  |
| Fully connected	666 	| Outputs 396        									|
| RELU					|												|
| Dropout				|	0.75						  		|
| Fully connected	396	| Outputs 128          									|
| RELU					|												|
| Dropout				|	0.75						  		|
| Fully connected	128	| Outputs 43          									|
| RELU					|												|
| Softmax				|      									|
| Cross entropy	|      									|
| Back prop			|      									|



#### Model Training

To train the model, I used the TensorFlow AdamOptimizer.  
```
EPOCHS = 100
BATCH_SIZE = 128
mu = 0
sigma = 0.1
rate = 0.0005
```

#### Solution Approach

First I tried the LeNet architecture from previous class but the result was below the 93% requirement. So next I tried another famous architecture AlexNet but a simplified version and achived 94%~95% validation accuracy.  

However, the image size in the provided dataset is only 32x32x3. And the images are very vague. We already lost a lot of information by downsizing the original images. And the over use of maxpooling will even lose more pixels and more information contained in those pixels. So eventually, I used convolution with VALID padding to reduce feature map size and only used maxpooling once after 5 convolution layers. After that I got a relatively large fully connected layer. To accelarate training and prevent overfit, I added the dropout function after each fully connected layer.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

 
---
### Test a Model on New Images

#### Acquiring New Images

I think five images is not enough to evaluate the system so I got 11. Here are they:  

![2](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/2.png)
![3](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/3.png)
![7](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/7.png)
![7_2](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/7_2.png)
![14](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/14.png)
![17](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/17.png)
![23](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/23.png)
![25](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/25.png)
![28](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/28.png)
![31](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/31.png)
![35](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/35.png)


The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


